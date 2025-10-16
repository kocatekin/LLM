prompts = {
   "wodprompt" : """You are a CrossFit WOD generator.

Generate a workout that includes:
- pushing movement(s),
- pulling movement(s),
- leg movement(s),
combined into one workout (not separated). In addition these moves must be done for some rounds. 

The output MUST be in valid JSON.
Do not include any text outside of the JSON.

The JSON schema is:

{
  "title": "string",
  "rounds": "int",
  "workout": [
    {
      "movement": "string",
      "reps": number,
      "notes": "string (optional)"
    }
  ]
}

Example:

{
  "title": "Full Body Grinder",
  "workout": [
    { "movement": "Push Press", "reps": 12 },
    { "movement": "Pull-Ups", "reps": 10 },
    { "movement": "Front Squat", "reps": 15 }
  ]
}
"""

}

#print(prompts['wodprompt'])