// Package llm provides a unified interface for interacting with various Language Learning Model providers.
package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/mail"
	"net/netip"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/go-playground/validator/v10"
	"github.com/teilomillet/gollm/utils"
)

// validate is the shared validator instance used across the package.
var validate *validator.Validate

// ValidatorFunc is the signature for custom validation functions.
// Custom validators receive the value to validate and return an error if validation fails.
// Use gollm.DefaultValidate within custom validators to fall back to standard validation.
type ValidatorFunc func(interface{}) error

// DefaultValidate performs the standard struct validation using the go-playground/validator.
// This is the canonical validation function that all other validation methods delegate to.
// Custom validators can call this to fall back to default validation behavior.
func DefaultValidate(v interface{}) error {
	return validate.Struct(v)
}

func init() {
	validate = validator.New()

	// Register custom validator for API key map
	if err := validate.RegisterValidation("apikey", validateAPIKey); err != nil {
		// Since this is in init(), we can't return the error
		// Instead, panic with a clear message as this is a critical setup failure
		panic(fmt.Sprintf("failed to register API key validator: %v", err))
	}
}

// validateAPIKey checks if the API key map contains a valid key for the current provider
func validateAPIKey(fl validator.FieldLevel) bool {
	apiKeys, ok := fl.Field().Interface().(map[string]string)
	if !ok {
		return false
	}

	// Get the parent struct (Config)
	parent := fl.Parent()
	provider := parent.FieldByName("Provider").String()

	// For local LLM servers, we don't require an API key.
	// Just check if the endpoint is accessible (or return true for vLLM).
	switch provider {
	case "vllm":
		// vLLM uses OpenAI-compatible API without authentication
		return true
	case "ollama":
		endpoint := parent.FieldByName("OllamaEndpoint").String()
		if endpoint == "" {
			endpoint = "http://localhost:11434"
		}
		resp, err := http.Head(endpoint + "/api/tags")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		return resp.StatusCode == http.StatusOK
	case "lmstudio":
		// LM Studio uses OpenAI-compatible API at localhost:1234
		resp, err := http.Get("http://localhost:1234/v1/models")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		return resp.StatusCode == http.StatusOK
	}

	// For other providers, check if there's a key for the provider
	apiKey, exists := apiKeys[provider]
	if !exists || apiKey == "" {
		return false
	}

	// Validate key format based on provider
	switch provider {
	case "openai":
		return strings.HasPrefix(apiKey, "sk-") && len(apiKey) > 20
	case "anthropic":
		return strings.HasPrefix(apiKey, "sk-ant-") && len(apiKey) > 20
	default:
		return len(apiKey) > 20 // Generic validation for unknown providers
	}
}

// Validate checks if the given struct is valid according to its validation rules.
// It delegates to DefaultValidate to maintain a single code path for validation.
//
// Parameters:
//   - s: The struct to validate. Must be a pointer to a struct.
//
// Returns:
//   - error: nil if validation passes, otherwise returns validation errors
//
// Example:
//
//	type Config struct {
//	    Model     string `validate:"required"`
//	    MaxTokens int    `validate:"min=1,max=4096"`
//	}
//
//	config := Config{Model: "gpt-4", MaxTokens: 2048}
//	if err := Validate(&config); err != nil {
//	    log.Fatal(err)
//	}
func Validate(s interface{}) error {
	return DefaultValidate(s)
}

// ValidateWithCustomValidator checks if the given struct is valid, using a custom validator if provided.
// This allows for scoped custom validation without affecting other goroutines.
//
// Parameters:
//   - s: The struct to validate. Must be a pointer to a struct.
//   - customValidator: Optional ValidatorFunc. If nil, uses default validation.
//
// Returns:
//   - error: nil if validation passes, otherwise returns validation errors
//
// Example:
//
//	// Skip validation for Google/Gemini provider
//	err := gollm.ValidateWithCustomValidator(cfg, func(v interface{}) error {
//	    if config, ok := v.(*gollm.Config); ok && config.Provider == "google" {
//	        return nil // Skip validation for Google
//	    }
//	    return gollm.DefaultValidate(v) // Use default for others
//	})
func ValidateWithCustomValidator(s interface{}, customValidator ValidatorFunc) error {
	if customValidator != nil {
		return customValidator(s)
	}
	return DefaultValidate(s)
}

// RegisterCustomValidation registers a custom validation function with the validator.
// This allows adding domain-specific validation rules beyond the standard ones.
//
// Parameters:
//   - tag: The validation tag to register (e.g., "customcheck")
//   - fn: The validation function to register
//
// Returns:
//   - error: nil if registration succeeds, otherwise returns an error
//
// Example:
//
//	func validateModel(fl validator.FieldLevel) bool {
//	    return strings.HasPrefix(fl.Field().String(), "gpt-")
//	}
//
//	err := RegisterCustomValidation("model", validateModel)
func RegisterCustomValidation(tag string, fn validator.Func) error {
	return validate.RegisterValidation(tag, fn)
}

// GenerateJSONSchema generates a JSON schema for the given struct.
// The schema includes type information, validation rules, and nested structures.
//
// Parameters:
//   - v: The struct to generate schema for
//
// Returns:
//   - []byte: The generated JSON schema
//   - error: Any error encountered during generation
//
// Example:
//
//	type Prompt struct {
//	    Text      string   `json:"text" validate:"required"`
//	    MaxTokens int      `json:"max_tokens" validate:"min=1"`
//	    Stop      []string `json:"stop,omitempty"`
//	}
//
//	schema, err := GenerateJSONSchema(&Prompt{})
func GenerateJSONSchema(v interface{}) ([]byte, error) {
	t := reflect.TypeOf(v)
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == nil || t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("GenerateJSONSchema needs a struct or a pointer to one, got %T", v)
	}

	schema := make(map[string]interface{})
	schema["type"] = "object"
	properties, required, err := getStructProperties(t)
	if err != nil {
		return nil, err
	}
	schema["properties"] = properties
	if len(required) > 0 {
		schema["required"] = required
	}
	return json.MarshalIndent(schema, "", "  ")
}

// getStructProperties analyzes a struct type and returns its JSON schema properties.
// It processes struct fields, their types, and validation rules to build the schema.
//
// Parameters:
//   - t: The reflect.Type of the struct to analyze
//
// Returns:
//   - map[string]interface{}: Schema properties
//   - []string: List of required fields
//   - error: Any error encountered during analysis
func getStructProperties(t reflect.Type) (map[string]interface{}, []string, error) {
	properties := make(map[string]interface{})
	var required []string

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" {
			continue
		}
		jsonName := strings.Split(jsonTag, ",")[0]
		if jsonName == "" {
			jsonName = field.Name
		}

		fieldSchema, err := getFieldSchema(field)
		if err != nil {
			return nil, nil, err
		}
		properties[jsonName] = fieldSchema

		if hasListRule(field.Tag.Get("validate"), "required") {
			required = append(required, jsonName)
		}
	}

	return properties, required, nil
}

// getFieldSchema generates a JSON schema for a single struct field.
// It handles various Go types and their corresponding JSON schema representations.
//
// Parameters:
//   - field: The reflect.StructField to generate schema for
//
// Returns:
//   - map[string]interface{}: Field schema
//   - error: Any error encountered during generation
func getFieldSchema(field reflect.StructField) (map[string]interface{}, error) {
	schema := make(map[string]interface{})

	switch field.Type.Kind() {
	case reflect.String:
		schema["type"] = "string"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		schema["type"] = "integer"
	case reflect.Float32, reflect.Float64:
		schema["type"] = "number"
	case reflect.Bool:
		schema["type"] = "boolean"
	case reflect.Slice:
		schema["type"] = "array"
		itemSchema, err := getFieldSchema(reflect.StructField{Type: field.Type.Elem()})
		if err != nil {
			return nil, err
		}
		schema["items"] = itemSchema
	case reflect.Struct:
		schema["type"] = "object"
		properties, required, err := getStructProperties(field.Type)
		if err != nil {
			return nil, err
		}
		schema["properties"] = properties
		if len(required) > 0 {
			schema["required"] = required
		}
	default:
		return nil, fmt.Errorf("unsupported type: %v", field.Type.Kind())
	}

	addValidationToSchema(schema, field.Tag.Get("validate"))

	return schema, nil
}

// addValidationToSchema adds validation rules from struct tags to the JSON schema.
// It converts Go validation rules to their JSON Schema equivalents. A rule a schema cannot hold in a
// form ValidateAgainstSchema enforces (excludes, password=strong's character classes, a pattern Go's
// regexp cannot compile, a second pattern) is written into the field's description instead, so the
// schema always passes ValidateSchema and the model still sees the rule. ExtractStructuredData
// enforces those rules with Validate on the parsed struct; GenerateWithSchema does not.
//
// Parameters:
//   - schema: The schema to add validation rules to
//   - validateTag: The validation tag string to process
func addValidationToSchema(schema map[string]interface{}, validateTag string) {
	rules := strings.Split(validateTag, ",")
	omitEmpty := false
	// Rules are added straight to schema; under omitempty they are moved into the description at the
	// end, since a schema cannot say "empty, or else at least 3 characters" without anyOf.
	defer func() {
		if omitEmpty {
			describeIfNotEmpty(schema)
		}
	}()
	for i, rule := range rules {
		parts := strings.SplitN(rule, "=", 2)
		key := parts[0]
		var value string
		if len(parts) > 1 {
			value = parts[1]
		}

		switch key {
		case "required":
			// This is handled in generateJSONSchemaFromStruct

		case "omitempty":
			// As in the validator package, an empty value skips the field's other rules.
			omitEmpty = true

		case "dive":
			// As in the validator package, the rules after dive apply to each item, not the list.
			if items, ok := schema["items"].(map[string]interface{}); ok {
				addValidationToSchema(items, strings.Join(rules[i+1:], ","))
			}
			return

		// min/gte, max/lte and len bound a string's length and an array's item count, as in the
		// validator package, and a number's value.
		case "min", "gte":
			if num, err := strconv.ParseFloat(value, 64); err == nil {
				switch schema["type"] {
				case "array":
					schema["minItems"] = int(num)
				case "string":
					schema["minLength"] = int(num)
				default:
					schema["minimum"] = num
				}
			}

		case "max", "lte":
			if num, err := strconv.ParseFloat(value, 64); err == nil {
				switch schema["type"] {
				case "array":
					schema["maxItems"] = int(num)
				case "string":
					schema["maxLength"] = int(num)
				default:
					schema["maximum"] = num
				}
			}

		case "len":
			if num, err := strconv.ParseInt(value, 10, 64); err == nil {
				switch schema["type"] {
				case "array":
					schema["minItems"] = num
					schema["maxItems"] = num
				case "string":
					schema["minLength"] = num
					schema["maxLength"] = num
				default:
					schema["minimum"] = num
					schema["maximum"] = num
				}
			}

		case "one_decimal":
			schema["multipleOf"] = 0.1

		case "email":
			schema["format"] = "email"

		case "url":
			schema["format"] = "uri"

		case "datetime":
			schema["format"] = "date-time"

		case "regex":
			setPattern(schema, value, fmt.Sprintf("Must match the regular expression %q.", value))

		case "enum":
			setEnum(schema, strings.Split(value, "|"))

		case "contains":
			// A JSON Schema pattern matches anywhere in the string, so the quoted text is enough.
			setPattern(schema, regexp.QuoteMeta(value), fmt.Sprintf("Must contain %q.", value))

		case "excludes":
			// "Does not contain" needs a negative lookahead, which Go's regexp lacks.
			appendDescription(schema, fmt.Sprintf("Must not contain %q.", value))

		case "unique":
			if value == "true" {
				schema["uniqueItems"] = true
			}

		case "minItems":
			if num, err := strconv.ParseInt(value, 10, 64); err == nil {
				schema["minItems"] = num
			}

		case "maxItems":
			if num, err := strconv.ParseInt(value, 10, 64); err == nil {
				schema["maxItems"] = num
			}

		case "password":
			// The allowed characters and the length fit one pattern; requiring one of each class takes
			// lookaheads, which Go's regexp lacks, so that part is described.
			setPattern(schema, `^[A-Za-z\d@$!%*?&]{8,}$`, "Must be at least 8 characters of letters, digits and @$!%*?&.")
			appendDescription(schema, "Must include an uppercase letter, a lowercase letter, a digit and one of @$!%*?&.")

			// Add more cases as needed
		}
	}
}

// hasListRule reports whether a validate tag holds rule itself, before any dive: "dive,required"
// requires each item, not the list, and "required_if" is a different rule.
func hasListRule(validateTag, rule string) bool {
	for _, r := range strings.Split(validateTag, ",") {
		key := strings.SplitN(r, "=", 2)[0]
		if key == "dive" {
			return false
		}
		if key == rule {
			return true
		}
	}
	return false
}

// constraintKeywords are the keywords addValidationToSchema adds that constrain a value. Under
// omitempty they move into the description.
var constraintKeywords = []string{
	"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
	"minLength", "maxLength", "pattern", "format", "enum",
	"minItems", "maxItems", "uniqueItems",
}

// describeIfNotEmpty moves a field's constraints into its description, qualified so an empty value
// still passes, as omitempty means in the validator package.
func describeIfNotEmpty(schema map[string]interface{}) {
	var sentences []string
	for _, k := range constraintKeywords {
		v, ok := schema[k]
		if !ok {
			continue
		}
		if k == "enum" {
			// DescribeSchemaRule renders a decoded list; the generator builds typed slices.
			raw, _ := json.Marshal(v)
			_ = json.Unmarshal(raw, &v)
		}
		if sentence, ok := utils.DescribeSchemaRule(k, v); ok {
			sentences = append(sentences, sentence)
		}
		delete(schema, k)
	}
	if len(sentences) > 0 {
		appendDescription(schema, "If not empty: "+strings.Join(sentences, " "))
	}
}

// setEnum sets a field's enum with values of the field's type, so enum=1|2 on an int matches the
// numbers a response holds. A value that does not parse as that type is described instead.
func setEnum(schema map[string]interface{}, values []string) {
	typed := make([]interface{}, 0, len(values))
	for _, v := range values {
		var parsed interface{} = v
		var err error
		switch schema["type"] {
		case "integer":
			parsed, err = strconv.ParseInt(v, 10, 64)
		case "number":
			parsed, err = strconv.ParseFloat(v, 64)
		case "boolean":
			parsed, err = strconv.ParseBool(v)
		}
		if err != nil {
			appendDescription(schema, fmt.Sprintf("Must be one of: %s.", strings.Join(values, ", ")))
			return
		}
		typed = append(typed, parsed)
	}
	schema["enum"] = typed
}

// setPattern sets a field's pattern. A schema holds one pattern per field, and the validator runs it
// with Go's regexp, so a second pattern, or one Go cannot compile, is described instead.
func setPattern(schema map[string]interface{}, pattern, description string) {
	if _, taken := schema["pattern"]; taken {
		appendDescription(schema, description)
		return
	}
	if _, err := regexp.Compile(pattern); err != nil {
		appendDescription(schema, description)
		return
	}
	schema["pattern"] = pattern
}

// appendDescription adds a sentence to a field's description.
func appendDescription(schema map[string]interface{}, sentence string) {
	if existing, ok := schema["description"].(string); ok && existing != "" {
		schema["description"] = existing + " " + sentence
		return
	}
	schema["description"] = sentence
}

// ValidateAgainstSchema validates a JSON response against a JSON schema.
// It ensures the response matches the expected structure and constraints.
//
// Parameters:
//   - response: The JSON response string to validate
//   - schema: The schema to validate against
//
// Returns:
//   - error: nil if validation passes, otherwise returns validation errors
//
// Example:
//
//	schema := map[string]interface{}{
//	    "type": "object",
//	    "properties": map[string]interface{}{
//	        "text": map[string]interface{}{
//	            "type": "string",
//	        },
//	    },
//	}
//
//	err := ValidateAgainstSchema(`{"text": "Hello"}`, schema)
func ValidateAgainstSchema(response string, schema interface{}) error {
	var responseData interface{}
	if err := json.Unmarshal([]byte(response), &responseData); err != nil {
		return fmt.Errorf("failed to parse response JSON: %w", err)
	}

	schemaMap, err := decodeSchema(schema)
	if err != nil {
		return err
	}
	if err := checkSchema(schemaMap); err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}

	if err := validateJSONAgainstSchema(responseData, schemaMap); err != nil {
		return fmt.Errorf("response does not match schema: %w", err)
	}

	return nil
}

// ValidateSchema reports whether ValidateAgainstSchema can check responses against a schema, without
// needing a response. It returns an error for a schema that would fail every response: no schema at
// all, a malformed or unknown "type", or a malformed "properties", "required" or "items". A node with
// no "type" accepts any value. GenerateWithSchema calls it before sending a request, so a broken
// schema fails at once instead of after every billed retry.
func ValidateSchema(schema interface{}) error {
	schemaMap, err := decodeSchema(schema)
	if err != nil {
		return err
	}
	if err := checkSchema(schemaMap); err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}
	return nil
}

// decodeSchema turns a schema given as a JSON string, JSON bytes, a Go map or a struct into the
// shapes JSON decodes to. Go map literals are round-tripped too, so a []string type list arrives as
// []interface{} and a map[string]string property schema as map[string]interface{}.
func decodeSchema(schema interface{}) (map[string]interface{}, error) {
	var schemaMap map[string]interface{}
	switch s := schema.(type) {
	case string:
		if err := json.Unmarshal([]byte(s), &schemaMap); err != nil {
			return nil, fmt.Errorf("failed to parse schema JSON string: %w", err)
		}
	case []byte:
		if err := json.Unmarshal(s, &schemaMap); err != nil {
			return nil, fmt.Errorf("failed to parse schema JSON bytes: %w", err)
		}
	default:
		schemaBytes, err := json.Marshal(schema)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal schema: %w", err)
		}
		if err := json.Unmarshal(schemaBytes, &schemaMap); err != nil {
			return nil, fmt.Errorf("failed to parse schema JSON: %w", err)
		}
	}
	if schemaMap == nil {
		return nil, fmt.Errorf("schema is null: want a JSON object")
	}
	return normalizeDraft4Bounds(schemaMap), nil
}

// normalizeDraft4Bounds rewrites draft-04's boolean exclusiveMinimum/exclusiveMaximum to the numeric
// form throughout a decoded schema, so the check and the validator read one form.
func normalizeDraft4Bounds(schema map[string]interface{}) map[string]interface{} {
	schema = utils.NormalizeExclusiveBounds(schema)
	if properties, ok := schema["properties"].(map[string]interface{}); ok {
		for key, prop := range properties {
			if propMap, ok := prop.(map[string]interface{}); ok {
				properties[key] = normalizeDraft4Bounds(propMap)
			}
		}
	}
	for _, keyword := range []string{"items", "additionalProperties"} {
		if sub, ok := schema[keyword].(map[string]interface{}); ok {
			schema[keyword] = normalizeDraft4Bounds(sub)
		}
	}
	return schema
}

// compiledPatterns caches each schema pattern's regexp, since one pattern is checked against every
// value under it, on every attempt.
var compiledPatterns sync.Map

// compilePattern compiles a schema pattern with Go's regexp, once per pattern.
func compilePattern(pattern string) (*regexp.Regexp, error) {
	if re, ok := compiledPatterns.Load(pattern); ok {
		return re.(*regexp.Regexp), nil
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, err
	}
	compiledPatterns.Store(pattern, re)
	return re, nil
}

// unsupportedSchemaKeywords combine or reference other schemas. The validator cannot evaluate them,
// and accepting a node that uses one would let any response through it unchecked, so checkSchema
// rejects them before a request is sent.
var unsupportedSchemaKeywords = []string{
	"anyOf", "oneOf", "allOf", "not", "if", "then", "else",
	"$ref", "$dynamicRef", "prefixItems", "contains", "patternProperties", "propertyNames",
	"dependentSchemas", "dependentRequired", "unevaluatedProperties", "unevaluatedItems",
}

// typeMatcher returns how to recognize a value of one JSON Schema type, and false for a name the
// validator does not know. It is the only list of supported type names.
func typeMatcher(name string) (func(data interface{}) bool, bool) {
	switch name {
	case "object":
		return func(d interface{}) bool { _, ok := d.(map[string]interface{}); return ok }, true
	case "array":
		return func(d interface{}) bool { _, ok := d.([]interface{}); return ok }, true
	case "string":
		return func(d interface{}) bool { _, ok := d.(string); return ok }, true
	case "number":
		return func(d interface{}) bool { _, ok := d.(float64); return ok }, true
	case "integer":
		return func(d interface{}) bool { f, ok := d.(float64); return ok && f == math.Trunc(f) }, true
	case "boolean":
		return func(d interface{}) bool { _, ok := d.(bool); return ok }, true
	case "null":
		return func(d interface{}) bool { return d == nil }, true
	}
	return nil, false
}

// checkSchema checks a decoded schema node and everything beneath it: every keyword the validator
// reads must be well formed, every listed type known, and no unsupported keyword present. A node with
// no "type" accepts any value, as in {"description": "free text"}, but its other keywords still apply.
func checkSchema(schema map[string]interface{}) error {
	for _, keyword := range unsupportedSchemaKeywords {
		if _, present := schema[keyword]; present {
			return fmt.Errorf("unsupported schema keyword %q: the validator cannot check it", keyword)
		}
	}

	if raw, present := schema["type"]; present {
		types, ok := utils.SchemaTypes(raw)
		if !ok {
			return fmt.Errorf("invalid 'type' in schema: %v", raw)
		}
		for _, name := range types {
			if _, ok := typeMatcher(name); !ok {
				return fmt.Errorf("unsupported schema type: %s", name)
			}
		}
	}

	if raw, present := schema["enum"]; present {
		if _, ok := raw.([]interface{}); !ok {
			return fmt.Errorf("invalid 'enum' in schema: want a list")
		}
	}
	for _, keyword := range []string{"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"} {
		if raw, present := schema[keyword]; present {
			n, ok := raw.(float64)
			if !ok || (keyword == "multipleOf" && n <= 0) {
				return fmt.Errorf("invalid '%s' in schema: %v", keyword, raw)
			}
		}
	}
	for _, keyword := range []string{"minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"} {
		if raw, present := schema[keyword]; present {
			n, ok := raw.(float64)
			if !ok || n < 0 || n != math.Trunc(n) {
				return fmt.Errorf("invalid '%s' in schema: %v", keyword, raw)
			}
		}
	}
	if raw, present := schema["uniqueItems"]; present {
		if _, ok := raw.(bool); !ok {
			return fmt.Errorf("invalid 'uniqueItems' in schema: %v", raw)
		}
	}
	if raw, present := schema["format"]; present {
		if _, ok := raw.(string); !ok {
			return fmt.Errorf("invalid 'format' in schema: %v", raw)
		}
	}
	if raw, present := schema["pattern"]; present {
		pattern, ok := raw.(string)
		if !ok {
			return fmt.Errorf("invalid 'pattern' in schema: %v", raw)
		}
		if _, err := compilePattern(pattern); err != nil {
			return fmt.Errorf("unsupported 'pattern' %q: %w", pattern, err)
		}
	}

	if raw, present := schema["properties"]; present {
		properties, ok := raw.(map[string]interface{})
		if !ok {
			return fmt.Errorf("invalid 'properties' in schema")
		}
		for key, propSchema := range properties {
			propMap, ok := propSchema.(map[string]interface{})
			if !ok {
				return fmt.Errorf("invalid schema for field '%s': got %T, want an object", key, propSchema)
			}
			if err := checkSchema(propMap); err != nil {
				return fmt.Errorf("invalid field '%s': %w", key, err)
			}
		}
	}
	if _, err := requiredFields(schema["required"]); err != nil {
		return err
	}
	if raw, present := schema["additionalProperties"]; present {
		switch ap := raw.(type) {
		case bool:
		case map[string]interface{}:
			if err := checkSchema(ap); err != nil {
				return fmt.Errorf("invalid additionalProperties: %w", err)
			}
		default:
			return fmt.Errorf("invalid 'additionalProperties' in schema: %v", raw)
		}
	}
	if raw, present := schema["items"]; present {
		items, ok := raw.(map[string]interface{})
		if !ok {
			return fmt.Errorf("invalid 'items' in schema")
		}
		if err := checkSchema(items); err != nil {
			return fmt.Errorf("invalid items: %w", err)
		}
	}
	return nil
}

// validateJSONAgainstSchema performs the actual JSON schema validation.
// It recursively validates complex data structures against their schema.
//
// The schema must be decoded and must have passed checkSchema, as ValidateAgainstSchema ensures, so
// the checks below read keywords without re-checking their shape. As in JSON Schema, a keyword
// applies to the kind of value it constrains whatever "type" says: "minimum" to numbers,
// "maxLength" to strings, "properties" to objects. "type", when present, must also match.
//
// Parameters:
//   - data: The data to validate
//   - schema: The schema to validate against
//
// Returns:
//   - error: nil if validation passes, otherwise returns validation errors
func validateJSONAgainstSchema(data interface{}, schema map[string]interface{}) error {
	// A type list (e.g. ["string", "null"], OpenAI strict mode's nullable) passes if any member does.
	if raw, present := schema["type"]; present {
		types, _ := utils.SchemaTypes(raw)
		matched := false
		for _, name := range types {
			if matches, ok := typeMatcher(name); ok && matches(data) {
				matched = true
				break
			}
		}
		if !matched {
			return fmt.Errorf("expected %s, got %s", strings.Join(types, " or "), jsonKind(data))
		}
	}

	if raw, present := schema["enum"]; present {
		allowed, _ := raw.([]interface{})
		found := false
		for _, v := range allowed {
			if reflect.DeepEqual(data, v) {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("value %v is not one of %v", data, allowed)
		}
	}
	if want, present := schema["const"]; present && !reflect.DeepEqual(data, want) {
		return fmt.Errorf("value %v is not %v", data, want)
	}

	switch d := data.(type) {
	case float64:
		return validateNumber(d, schema)
	case string:
		return validateString(d, schema)
	case []interface{}:
		return validateArray(d, schema)
	case map[string]interface{}:
		return validateObject(d, schema)
	}
	return nil
}

// jsonKind names the JSON kind of a decoded value for error messages.
func jsonKind(data interface{}) string {
	switch d := data.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case float64:
		if d == math.Trunc(d) {
			return "integer"
		}
		return "number"
	case string:
		return "string"
	case []interface{}:
		return "array"
	case map[string]interface{}:
		return "object"
	}
	return fmt.Sprintf("%T", data)
}

// requiredFields reads "required" as a list of field names, from a JSON-decoded schema. A value that
// is not a list, such as the old draft-3 "required": true on a property, is ignored as it always has been.
func requiredFields(r interface{}) ([]string, error) {
	list, ok := r.([]interface{})
	if !ok {
		return nil, nil
	}
	fields := make([]string, 0, len(list))
	for _, item := range list {
		s, ok := item.(string)
		if !ok {
			return nil, fmt.Errorf("schema 'required' list holds %T, want string", item)
		}
		fields = append(fields, s)
	}
	return fields, nil
}

// validateNumber checks a number against the numeric keywords.
func validateNumber(n float64, schema map[string]interface{}) error {
	if limit, ok := schema["minimum"].(float64); ok && n < limit {
		return fmt.Errorf("%v is less than minimum %v", n, limit)
	}
	if limit, ok := schema["maximum"].(float64); ok && n > limit {
		return fmt.Errorf("%v is greater than maximum %v", n, limit)
	}
	if limit, ok := schema["exclusiveMinimum"].(float64); ok && n <= limit {
		return fmt.Errorf("%v is not greater than exclusiveMinimum %v", n, limit)
	}
	if limit, ok := schema["exclusiveMaximum"].(float64); ok && n >= limit {
		return fmt.Errorf("%v is not less than exclusiveMaximum %v", n, limit)
	}
	if step, ok := schema["multipleOf"].(float64); ok {
		// A tolerance, so 0.3 counts as a multiple of 0.1 despite binary rounding.
		q := n / step
		if math.Abs(q-math.Round(q)) > 1e-9*math.Max(1, math.Abs(q)) {
			return fmt.Errorf("%v is not a multiple of %v", n, step)
		}
	}
	return nil
}

// uuidPattern matches a UUID in its canonical 8-4-4-4-12 hex form.
var uuidPattern = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`)

// formatCheckers check the "format" values the providers document for strict schemas. Any other
// format is an annotation and is not checked, as JSON Schema leaves it.
var formatCheckers = map[string]func(string) bool{
	"date-time": func(s string) bool { _, err := time.Parse(time.RFC3339Nano, s); return err == nil },
	"date":      func(s string) bool { _, err := time.Parse("2006-01-02", s); return err == nil },
	"time": func(s string) bool {
		for _, layout := range []string{"15:04:05.999999999Z07:00", "15:04:05.999999999"} {
			if _, err := time.Parse(layout, s); err == nil {
				return true
			}
		}
		return false
	},
	"email": func(s string) bool {
		addr, err := mail.ParseAddress(s)
		return err == nil && addr.Name == "" && addr.Address == s
	},
	"uuid": uuidPattern.MatchString,
	"ipv4": func(s string) bool { a, err := netip.ParseAddr(s); return err == nil && a.Is4() },
	"ipv6": func(s string) bool { a, err := netip.ParseAddr(s); return err == nil && a.Is6() },
}

// validateString checks a string against the string keywords. Lengths count Unicode code points,
// as JSON Schema does.
func validateString(s string, schema map[string]interface{}) error {
	length := float64(utf8.RuneCountInString(s))
	if limit, ok := schema["minLength"].(float64); ok && length < limit {
		return fmt.Errorf("length %v is less than minLength %v", length, limit)
	}
	if limit, ok := schema["maxLength"].(float64); ok && length > limit {
		return fmt.Errorf("length %v is greater than maxLength %v", length, limit)
	}
	if pattern, ok := schema["pattern"].(string); ok {
		re, err := compilePattern(pattern)
		if err != nil || !re.MatchString(s) {
			return fmt.Errorf("%q does not match pattern %q", s, pattern)
		}
	}
	if format, ok := schema["format"].(string); ok {
		if check, known := formatCheckers[format]; known && !check(s) {
			return fmt.Errorf("%q is not a valid %s", s, format)
		}
	}
	return nil
}

// validateArray checks an array against the array keywords, then each item against "items". No
// "items" means any items, as in {"type": ["array", "null"]}.
func validateArray(list []interface{}, schema map[string]interface{}) error {
	count := float64(len(list))
	if limit, ok := schema["minItems"].(float64); ok && count < limit {
		return fmt.Errorf("%v items is fewer than minItems %v", count, limit)
	}
	if limit, ok := schema["maxItems"].(float64); ok && count > limit {
		return fmt.Errorf("%v items is more than maxItems %v", count, limit)
	}
	if unique, _ := schema["uniqueItems"].(bool); unique {
		for i := range list {
			for j := i + 1; j < len(list); j++ {
				if reflect.DeepEqual(list[i], list[j]) {
					return fmt.Errorf("items at index %d and %d are equal, but uniqueItems is set", i, j)
				}
			}
		}
	}

	items, present := schema["items"].(map[string]interface{})
	if !present {
		return nil
	}
	for i, item := range list {
		if err := validateJSONAgainstSchema(item, items); err != nil {
			return fmt.Errorf("invalid item at index %d: %w", i, err)
		}
	}
	return nil
}

// validateObject checks an object against the object keywords: required fields, each listed property,
// and fields beyond "properties", which additionalProperties false forbids and an additionalProperties
// schema constrains. No "properties" means any fields, as in {"type": ["object", "null"]}.
func validateObject(object map[string]interface{}, schema map[string]interface{}) error {
	count := float64(len(object))
	if limit, ok := schema["minProperties"].(float64); ok && count < limit {
		return fmt.Errorf("%v fields is fewer than minProperties %v", count, limit)
	}
	if limit, ok := schema["maxProperties"].(float64); ok && count > limit {
		return fmt.Errorf("%v fields is more than maxProperties %v", count, limit)
	}

	// Required fields are checked on their own, so one missing from "properties" (or with no
	// "properties" at all) is still enforced.
	required, _ := requiredFields(schema["required"])
	for _, key := range required {
		if _, exists := object[key]; !exists {
			return fmt.Errorf("missing required field: %s", key)
		}
	}

	properties, _ := schema["properties"].(map[string]interface{})
	for key, propSchema := range properties {
		propData, exists := object[key]
		if !exists {
			continue
		}
		propMap, _ := propSchema.(map[string]interface{})
		if err := validateJSONAgainstSchema(propData, propMap); err != nil {
			return fmt.Errorf("invalid field '%s': %w", key, err)
		}
	}

	switch extra := schema["additionalProperties"].(type) {
	case bool:
		if !extra {
			for key := range object {
				if _, listed := properties[key]; !listed {
					return fmt.Errorf("unexpected field '%s': additionalProperties is false", key)
				}
			}
		}
	case map[string]interface{}:
		for key, value := range object {
			if _, listed := properties[key]; listed {
				continue
			}
			if err := validateJSONAgainstSchema(value, extra); err != nil {
				return fmt.Errorf("invalid field '%s': %w", key, err)
			}
		}
	}
	return nil
}
