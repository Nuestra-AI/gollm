package llm

import (
	"errors"
	"sync"
	"testing"
)

// testStruct is a simple struct for validation testing
type testStruct struct {
	Name  string `validate:"required"`
	Value int    `validate:"min=1,max=100"`
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		wantErr bool
	}{
		{
			name:    "valid struct",
			input:   &testStruct{Name: "test", Value: 50},
			wantErr: false,
		},
		{
			name:    "missing required field",
			input:   &testStruct{Name: "", Value: 50},
			wantErr: true,
		},
		{
			name:    "value below minimum",
			input:   &testStruct{Name: "test", Value: 0},
			wantErr: true,
		},
		{
			name:    "value above maximum",
			input:   &testStruct{Name: "test", Value: 101},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Validate(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateWithCustomValidator(t *testing.T) {
	t.Run("nil custom validator uses default", func(t *testing.T) {
		valid := &testStruct{Name: "test", Value: 50}
		invalid := &testStruct{Name: "", Value: 50}

		if err := ValidateWithCustomValidator(valid, nil); err != nil {
			t.Errorf("expected no error for valid struct, got %v", err)
		}

		if err := ValidateWithCustomValidator(invalid, nil); err == nil {
			t.Error("expected error for invalid struct, got nil")
		}
	})

	t.Run("custom validator that always passes", func(t *testing.T) {
		invalid := &testStruct{Name: "", Value: 0}
		alwaysPass := func(v interface{}) error { return nil }

		if err := ValidateWithCustomValidator(invalid, alwaysPass); err != nil {
			t.Errorf("expected custom validator to pass, got %v", err)
		}
	})

	t.Run("custom validator that always fails", func(t *testing.T) {
		valid := &testStruct{Name: "test", Value: 50}
		customErr := errors.New("custom validation error")
		alwaysFail := func(v interface{}) error { return customErr }

		err := ValidateWithCustomValidator(valid, alwaysFail)
		if err != customErr {
			t.Errorf("expected custom error, got %v", err)
		}
	})

	t.Run("custom validator with fallback to default", func(t *testing.T) {
		valid := &testStruct{Name: "test", Value: 50}
		invalid := &testStruct{Name: "", Value: 50}

		// Custom validator that only validates Name field, falls back for rest
		customValidator := func(v interface{}) error {
			if ts, ok := v.(*testStruct); ok {
				if ts.Name == "skip" {
					return nil // Skip validation for specific name
				}
			}
			return DefaultValidate(v)
		}

		// Valid struct should pass
		if err := ValidateWithCustomValidator(valid, customValidator); err != nil {
			t.Errorf("expected valid struct to pass, got %v", err)
		}

		// Invalid struct should fail (fallback to default)
		if err := ValidateWithCustomValidator(invalid, customValidator); err == nil {
			t.Error("expected invalid struct to fail")
		}

		// Struct with "skip" name should pass even if otherwise invalid
		skipStruct := &testStruct{Name: "skip", Value: 0}
		if err := ValidateWithCustomValidator(skipStruct, customValidator); err != nil {
			t.Errorf("expected 'skip' struct to pass, got %v", err)
		}
	})
}

func TestValidateWithCustomValidatorConcurrency(t *testing.T) {
	// This test verifies that custom validators are scoped and don't interfere
	// with each other when called concurrently from different goroutines.
	const numGoroutines = 100
	var wg sync.WaitGroup
	errCh := make(chan error, numGoroutines*2)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(2)

		// Goroutine with custom validator that always passes
		go func() {
			defer wg.Done()
			invalid := &testStruct{Name: "", Value: 0}
			alwaysPass := func(v interface{}) error { return nil }

			err := ValidateWithCustomValidator(invalid, alwaysPass)
			if err != nil {
				errCh <- errors.New("alwaysPass validator should have passed")
			}
		}()

		// Goroutine with nil validator (uses default)
		go func() {
			defer wg.Done()
			invalid := &testStruct{Name: "", Value: 0}

			err := ValidateWithCustomValidator(invalid, nil)
			if err == nil {
				errCh <- errors.New("nil validator should have failed for invalid struct")
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Error(err)
	}
}

func TestDefaultValidate(t *testing.T) {
	valid := &testStruct{Name: "test", Value: 50}
	invalid := &testStruct{Name: "", Value: 0}

	if err := DefaultValidate(valid); err != nil {
		t.Errorf("expected valid struct to pass, got %v", err)
	}

	if err := DefaultValidate(invalid); err == nil {
		t.Error("expected invalid struct to fail")
	}
}

// Schemas built as Go maps carry []string for a type list; JSON-decoded ones carry []interface{}.
func TestValidateAgainstSchemaTypeLists(t *testing.T) {
	nullableQuestion := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"question": map[string]interface{}{"type": []string{"string", "null"}},
		},
	}
	findings := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"findings": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"suggested_answer": map[string]interface{}{"type": []string{"string", "null"}},
					},
				},
			},
		},
	}

	tests := []struct {
		name     string
		response string
		schema   interface{}
		wantErr  bool
	}{
		{"string in nullable", `{"question": "Do you do metal roofs?"}`, nullableQuestion, false},
		{"null in nullable", `{"question": null}`, nullableQuestion, false},
		{"wrong type in nullable", `{"question": 5}`, nullableQuestion, true},
		{"nested nullable string", `{"findings": [{"suggested_answer": "a"}]}`, findings, false},
		{"nested nullable null", `{"findings": [{"suggested_answer": null}]}`, findings, false},
		{"type list from JSON", `{"q": null}`, `{"type":"object","properties":{"q":{"type":["string","null"]}}}`, false},
		{"null where not listed", `{"q": null}`, `{"type":"object","properties":{"q":{"type":"string"}}}`, true},
		{"non-string in type list", `{"q": "x"}`, `{"type":"object","properties":{"q":{"type":["string",1]}}}`, true},
		{"empty type list", `{"q": "x"}`, `{"type":"object","properties":{"q":{"type":[]}}}`, true},
		// A misspelled name is a schema error even when another listed type matches the data.
		{"misspelled name beside null", `{"q": null}`, `{"type":"object","properties":{"q":{"type":["strng","null"]}}}`, true},
		{"misspelled name in Go list", `{"q": null}`, map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{"q": map[string]interface{}{"type": []string{"strng", "null"}}},
		}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateAgainstSchema(tt.response, tt.schema)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateAgainstSchema() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// "required" is []string when the schema is built in Go, and an object schema with no "properties"
// accepts any object, which is how a free-form nullable object is written.
func TestValidateAgainstSchemaObjectKeywords(t *testing.T) {
	requiredGo := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"question": map[string]interface{}{"type": []string{"string", "null"}},
		},
		"required": []string{"question"},
	}
	freeForm := map[string]interface{}{"type": []string{"object", "null"}}

	tests := []struct {
		name     string
		response string
		schema   interface{}
		wantErr  bool
	}{
		{"required Go list, field missing", `{}`, requiredGo, true},
		{"required Go list, field null", `{"question": null}`, requiredGo, false},
		{"required JSON list, field missing", `{}`, `{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`, true},
		{"required list holds non-string", `{}`, `{"type":"object","properties":{"q":{"type":"string"}},"required":[1]}`, true},
		{"free-form nullable object, object", `{"k": 1}`, freeForm, false},
		{"free-form nullable object, null", `null`, freeForm, false},
		{"free-form nullable object, string", `"x"`, freeForm, true},
		{"properties of the wrong type", `{}`, `{"type":"object","properties":[]}`, true},
		{"required with no properties, field missing", `{}`, `{"type":"object","required":["id"]}`, true},
		{"required field not in properties, missing", `{"q": "x"}`, `{"type":"object","properties":{"q":{"type":"string"}},"required":["q","id"]}`, true},
		{"draft-3 required: true is ignored", `{"a": {}}`, `{"type":"object","properties":{"a":{"type":"object","properties":{},"required":true}}}`, false},
		{"free-form nullable array, array", `[1, "x"]`, `{"type":["array","null"]}`, false},
		{"free-form nullable array, string", `"x"`, `{"type":["array","null"]}`, true},
		{"property schema as map[string]string", `{"q": 5}`, map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{"q": map[string]string{"type": "string"}},
		}, true},
		{"property schema that is not an object", `{"q": 5}`, `{"type":"object","properties":{"q":true}}`, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateAgainstSchema(tt.response, tt.schema)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateAgainstSchema() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
