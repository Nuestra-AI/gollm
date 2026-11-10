// File: utils/schema.go
package utils

import (
	"fmt"
	"strings"
)

// SchemaTypes reads a JSON Schema "type" value as one type name or a list of them, such as
// ["string", "null"], which is how a nullable field is written. A list is []string when the schema
// is built in Go and []interface{} when it is decoded from JSON. It reports false for a missing or
// empty value, or a list holding anything but strings.
func SchemaTypes(t interface{}) ([]string, bool) {
	switch v := t.(type) {
	case string:
		return []string{v}, true
	case []string:
		return v, len(v) > 0
	case []interface{}:
		types := make([]string, 0, len(v))
		for _, item := range v {
			s, ok := item.(string)
			if !ok {
				return nil, false
			}
			types = append(types, s)
		}
		return types, len(types) > 0
	}
	return nil, false
}

// DescribeSchemaRule renders a constraint keyword as a sentence for a field's description, for a
// rule that cannot be sent as a keyword but should still reach the model. It reports false for a
// keyword that is not a constraint, or a value it cannot render.
func DescribeSchemaRule(keyword string, value interface{}) (string, bool) {
	switch keyword {
	case "minimum":
		return fmt.Sprintf("At least %v.", value), true
	case "maximum":
		return fmt.Sprintf("At most %v.", value), true
	case "exclusiveMinimum":
		return fmt.Sprintf("Greater than %v.", value), true
	case "exclusiveMaximum":
		return fmt.Sprintf("Less than %v.", value), true
	case "multipleOf":
		return fmt.Sprintf("A multiple of %v.", value), true
	case "minLength":
		return fmt.Sprintf("At least %v characters.", value), true
	case "maxLength":
		return fmt.Sprintf("At most %v characters.", value), true
	case "pattern":
		return fmt.Sprintf("Must match the regular expression %q.", value), true
	case "format":
		return fmt.Sprintf("Must be a valid %v.", value), true
	case "minItems":
		return fmt.Sprintf("At least %v items.", value), true
	case "maxItems":
		return fmt.Sprintf("At most %v items.", value), true
	case "uniqueItems":
		if unique, _ := value.(bool); unique {
			return "Items must be unique.", true
		}
	case "minProperties":
		return fmt.Sprintf("At least %v fields.", value), true
	case "maxProperties":
		return fmt.Sprintf("At most %v fields.", value), true
	case "const":
		return fmt.Sprintf("Must be %v.", value), true
	case "enum":
		if values, ok := value.([]interface{}); ok {
			parts := make([]string, len(values))
			for i, v := range values {
				parts[i] = fmt.Sprintf("%v", v)
			}
			return "Must be one of: " + strings.Join(parts, ", ") + ".", true
		}
	}
	return "", false
}

// NormalizeExclusiveBounds rewrites JSON Schema draft-04's boolean exclusiveMinimum and
// exclusiveMaximum, which modify minimum and maximum, into the numeric form later drafts use:
// {"minimum": 0, "exclusiveMinimum": true} becomes {"exclusiveMinimum": 0}. It returns the node
// unchanged when there is nothing to rewrite, and a rewritten copy otherwise.
func NormalizeExclusiveBounds(schema map[string]interface{}) map[string]interface{} {
	rewrite := func(out map[string]interface{}, exclusive, bound string) {
		flag, isBool := out[exclusive].(bool)
		if !isBool {
			return
		}
		delete(out, exclusive)
		if flag {
			if limit, ok := out[bound]; ok {
				out[exclusive] = limit
				delete(out, bound)
			}
		}
	}
	_, minBool := schema["exclusiveMinimum"].(bool)
	_, maxBool := schema["exclusiveMaximum"].(bool)
	if !minBool && !maxBool {
		return schema
	}
	out := make(map[string]interface{}, len(schema))
	for k, v := range schema {
		out[k] = v
	}
	rewrite(out, "exclusiveMinimum", "minimum")
	rewrite(out, "exclusiveMaximum", "maximum")
	return out
}
