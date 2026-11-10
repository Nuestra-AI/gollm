package utils

import (
	"reflect"
	"testing"
)

// A type is one name or a list, built in Go ([]string) or decoded from JSON ([]interface{}).
func TestSchemaTypes(t *testing.T) {
	tests := []struct {
		name   string
		in     interface{}
		want   []string
		wantOK bool
	}{
		{"single name", "string", []string{"string"}, true},
		{"Go list", []string{"object", "null"}, []string{"object", "null"}, true},
		{"JSON list", []interface{}{"string", "null"}, []string{"string", "null"}, true},
		{"missing", nil, nil, false},
		{"empty Go list", []string{}, nil, false},
		{"empty JSON list", []interface{}{}, nil, false},
		{"non-string in list", []interface{}{"string", 1.0}, nil, false},
		{"wrong type", 5.0, nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := SchemaTypes(tt.in)
			if ok != tt.wantOK || (tt.wantOK && !reflect.DeepEqual(got, tt.want)) {
				t.Errorf("SchemaTypes(%v) = %v, %v; want %v, %v", tt.in, got, ok, tt.want, tt.wantOK)
			}
		})
	}
}
