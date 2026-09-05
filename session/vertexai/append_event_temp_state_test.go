// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package vertexai

import (
	"context"
	"maps"
	"strings"
	"testing"

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/option"
	"google.golang.org/grpc"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// testReasoningEngine is a numeric reasoning engine ID. The client derives the
// session resource name from AppName, so it has to parse as one.
const testReasoningEngine = "5576569044451983360"

// newCapturingService builds a session service whose unary RPCs are
// short-circuited by an interceptor recording the outbound
// AppendEventRequest. Nothing leaves the process, so no network, no
// credentials and no replay fixture are involved.
func newCapturingService(t *testing.T) (session.Service, func() *aiplatformpb.AppendEventRequest) {
	t.Helper()

	var captured *aiplatformpb.AppendEventRequest
	intercept := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if r, ok := req.(*aiplatformpb.AppendEventRequest); ok {
			captured = r
		}
		// Deliberately never call invoker.
		return nil
	}

	svc, err := NewSessionService(t.Context(), VertexAIServiceConfig{
		ProjectID: "test-project",
		Location:  "us-central1",
	},
		option.WithoutAuthentication(),
		option.WithGRPCDialOption(grpc.WithUnaryInterceptor(intercept)),
	)
	if err != nil {
		t.Fatalf("NewSessionService failed: %v", err)
	}
	return svc, func() *aiplatformpb.AppendEventRequest { return captured }
}

// wireDelta is the state delta on the column-backed EventActions field.
func wireDelta(t *testing.T, req *aiplatformpb.AppendEventRequest) map[string]any {
	t.Helper()
	return req.GetEvent().GetActions().GetStateDelta().AsMap()
}

// rawEventDelta is the state delta inside the raw_event blob, the second
// serialization path out of the client. It is nil unless the event needed
// raw_event at all.
func rawEventDelta(t *testing.T, req *aiplatformpb.AppendEventRequest) map[string]any {
	t.Helper()
	raw := req.GetEvent().GetRawEvent()
	if raw == nil {
		return nil
	}
	actions, _ := raw.AsMap()["actions"].(map[string]any)
	delta, _ := actions["stateDelta"].(map[string]any)
	return delta
}

func assertNoTempKeys(t *testing.T, where string, delta map[string]any) {
	t.Helper()
	for k := range delta {
		if strings.HasPrefix(k, session.KeyPrefixTemp) {
			t.Errorf("temp key %q reached %s: %v", k, where, delta)
		}
	}
}

// TestAppendEventTempState pins that no temp: key reaches the outbound
// AppendEventRequest, on either serialization path, while staying readable
// in-process for the rest of the invocation.
func TestAppendEventTempState(t *testing.T) {
	t.Parallel()

	mixed := func() map[string]any { return map[string]any{"temp:k1": "v1", "sk": "v2"} }

	tests := []struct {
		name           string
		delta          map[string]any
		isolationScope string
		partial        bool
		wantNoRequest  bool
		wantNilActions bool
		wantDelta      map[string]any
		check          func(t *testing.T, sess *localSession, ev *session.Event)
	}{
		{
			name:      "happy_path",
			delta:     map[string]any{"sk": "v2"},
			wantDelta: map[string]any{"sk": "v2"},
		},
		{
			name:      "wire_delta",
			delta:     mixed(),
			wantDelta: map[string]any{"sk": "v2"},
		},
		{
			// Trimming empties the delta, and with no artifact delta either
			// createAiplatformpbEventActions returns nil actions.
			name:           "only_temp_keys",
			delta:          map[string]any{"temp:a": 1, "temp:b": 2},
			wantNilActions: true,
		},
		{
			name:           "empty_delta",
			delta:          map[string]any{},
			wantNilActions: true,
		},
		{
			name:           "nil_delta",
			delta:          nil,
			wantNilActions: true,
		},
		{
			// HasPrefix, not a longer-than-prefix check.
			name:           "key_is_exactly_the_prefix",
			delta:          map[string]any{"temp:": "v"},
			wantNilActions: true,
		},
		{
			// The prefix is exact and case sensitive.
			name:      "near_miss_keys",
			delta:     map[string]any{"temp": "v", "tempx": "v", "TEMP:k": "v"},
			wantDelta: map[string]any{"temp": "v", "tempx": "v", "TEMP:k": "v"},
		},
		{
			// Only top-level keys are invocation scoped.
			name:      "temp_nested_in_a_value",
			delta:     map[string]any{"sk": map[string]any{"temp:x": 1}},
			wantDelta: map[string]any{"sk": map[string]any{"temp:x": float64(1)}},
		},
		{
			name:      "other_prefixes_routed",
			delta:     map[string]any{"app:a": 1, "user:u": 2, "temp:t": 3},
			wantDelta: map[string]any{"app:a": float64(1), "user:u": float64(2)},
		},
		{
			// IsolationScope forces eventNeedsRawEvent, which json.Marshals the
			// whole event. A trim inside createAiplatformpbEventActions would
			// not have closed this path.
			name:           "raw_event",
			delta:          mixed(),
			isolationScope: "scope-a",
			wantDelta:      map[string]any{"sk": "v2"},
		},
		{
			name:          "partial_event",
			delta:         mixed(),
			partial:       true,
			wantNoRequest: true,
		},
		{
			name:      "input_not_mutated",
			delta:     mixed(),
			wantDelta: map[string]any{"sk": "v2"},
			check: func(t *testing.T, _ *localSession, ev *session.Event) {
				t.Helper()
				if _, ok := ev.Actions.StateDelta["temp:k1"]; !ok {
					t.Errorf("caller event lost temp:k1: %v", ev.Actions.StateDelta)
				}
			},
		},
		{
			name:      "in_process_visibility",
			delta:     mixed(),
			wantDelta: map[string]any{"sk": "v2"},
			check: func(t *testing.T, sess *localSession, _ *session.Event) {
				t.Helper()
				got, err := sess.State().Get("temp:k1")
				if err != nil {
					t.Errorf("temp:k1 not readable in-process after AppendEvent: %v", err)
					return
				}
				if got != "v1" {
					t.Errorf("sess.State().Get(%q) = %v, want %q", "temp:k1", got, "v1")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			svc, captured := newCapturingService(t)
			sess := &localSession{
				appName:   testReasoningEngine,
				userID:    "u1",
				sessionID: "sess-1",
				state:     map[string]any{},
			}
			ev := &session.Event{
				LLMResponse:    model.LLMResponse{Partial: tt.partial},
				ID:             "event1",
				Author:         "user",
				InvocationID:   "inv1",
				IsolationScope: tt.isolationScope,
				Actions:        session.EventActions{StateDelta: tt.delta},
			}
			wantInput := maps.Clone(tt.delta)

			if err := svc.AppendEvent(t.Context(), sess, ev); err != nil {
				t.Fatalf("AppendEvent failed: %v", err)
			}

			req := captured()
			if tt.wantNoRequest {
				if req != nil {
					t.Fatalf("expected no RPC, got AppendEventRequest %v", req)
				}
				return
			}
			if req == nil {
				t.Fatal("no AppendEventRequest was captured")
			}

			if tt.wantNilActions {
				if got := req.GetEvent().GetActions(); got != nil {
					t.Errorf("event.actions = %v, want nil", got)
				}
			} else if diff := cmp.Diff(tt.wantDelta, wireDelta(t, req)); diff != "" {
				t.Errorf("event.actions.state_delta mismatch (-want +got):\n%s", diff)
			}
			assertNoTempKeys(t, "actions.state_delta", wireDelta(t, req))
			assertNoTempKeys(t, "raw_event.actions.stateDelta", rawEventDelta(t, req))

			if tt.isolationScope != "" {
				if diff := cmp.Diff(tt.wantDelta, rawEventDelta(t, req)); diff != "" {
					t.Errorf("raw_event.actions.stateDelta mismatch (-want +got):\n%s", diff)
				}
			}

			// trimTempDeltaState copies rather than mutates, so the caller's
			// event must come back untouched.
			if diff := cmp.Diff(wantInput, ev.Actions.StateDelta); diff != "" {
				t.Errorf("caller event was mutated (-before +after):\n%s", diff)
			}

			if tt.check != nil {
				tt.check(t, sess, ev)
			}
		})
	}
}

// TestAppendEventContextCanceled pins that a cancelled context aborts before
// anything is put on the wire.
func TestAppendEventContextCanceled(t *testing.T) {
	t.Parallel()

	svc, captured := newCapturingService(t)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	sess := &localSession{
		appName:   testReasoningEngine,
		userID:    "u1",
		sessionID: "sess-1",
		state:     map[string]any{},
	}
	err := svc.AppendEvent(ctx, sess, &session.Event{
		ID:           "event1",
		Author:       "user",
		InvocationID: "inv1",
		Actions:      session.EventActions{StateDelta: map[string]any{"temp:k1": "v1", "sk": "v2"}},
	})
	if err == nil {
		t.Fatal("AppendEvent with a cancelled context returned nil error")
	}
	if !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("AppendEvent error = %v, want it to mention context canceled", err)
	}
	if req := captured(); req != nil {
		t.Errorf("an AppendEventRequest was captured despite the cancelled context: %v", req)
	}
}
