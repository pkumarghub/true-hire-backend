from __future__ import annotations

from src.utils.llm.gemini_llm import get_gemini_llm

candidate_key_params = {
    "skills": "comma separated  string technologies, programming languages, frameworks required or preferred (e.g., React.js, Node.js, Python, Java, AWS, Docker)",
    "years_experience": "How many years the candidate should have worked in total or relevant experience",
    "location": "Where the candidate is expected to work or should be located (e.g., Noida, Pune, Remote)",
    "degree_level": "Academic degrees or qualifications required (e.g., B.Tech, MBA, Masters, Bachelor)",
    "employment_type": "Type of employment (e.g., Permanent, Contractual, Full-time, Part-time)"
}


def extract_parameters_with_llm(jd_text: str, key_params: dict) -> dict:
    """
    Extract key parameters from job description text using Gemini LLM.

    Args:
        jd_text (str): The job description text to analyze

    Returns:
        dict: Extracted parameters as key-value pairs
    """
    import json

    prompt = f"Extract the following parameters from the description below. " \
        f"Return as JSON with each key and value. Description for each key:\n"

    # Add each parameter description to the prompt
    for k, v in key_params.items():
        prompt += f"- {k}: {v}\n"

    # Add the job description to analyze
    prompt += f"\nJob Description:\n{jd_text}\n\n" \
        f"Return ONLY a valid JSON object with the extracted parameters. " \
        f"If a parameter is not found, use an empty string or appropriate default value."

    try:
        # Get the Gemini LLM instance
        llm = get_gemini_llm()
        response = llm.invoke(prompt)
        response_text = response.content
        if not response_text or not response_text.strip():
            print("Empty response from LLM")
            return {k: "" for k in key_params}

        try:
            params = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                try:
                    params = json.loads(json_match.group(1))
                except:
                    print(
                        f"Failed to parse JSON even after extraction: {response_text}")
                    return {k: "" for k in key_params}
            else:
                print(f"No JSON found in response: {response_text}")
                return {k: "" for k in key_params}

        for key in key_params:
            if key not in params:
                params[key] = ""

        return params

    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response as JSON: {e}")
        return {k: "" for k in key_params}

    except Exception as e:
        print(f"Error extracting parameters with LLM: {e}")
        return {k: "" for k in key_params}
