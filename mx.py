input_string = ('Here is a summary of the two case summaries:\n\n'
                'Case 1: Unwelcome Holiday Souvenir\n\n'
                'A 15-year-old boy developed an itchy, red rash after getting a beach "henna tattoo" in Thailand. '
                'The likely cause is an allergic reaction to paraphenylenediamine (PPD), a chemical in the tattoo dye. '
                'The reaction will settle with a topical steroid, but the boy is now sensitized to PPD and may react '
                'to other products containing the chemical, such as hair dye, sunscreen, or medications.\n\n'
                'Case 2: Streaky Summer Rash\n\n'
                'A 33-year-old man developed a phytophotodermatitis rash after using a strimmer, which spattered phototoxic sap onto his skin. '
                'The rash resulted in redness and blistering and was caused by exposure to sunlight. With photoprotection, the rash will heal, '
                'but may leave post-inflammatory pigmentation for several months.')

# Parsing and formatting
def format_case_summary(input_string):
    # Split into sections by double newlines
    sections = input_string.split("\n\n")
    # output = f"**{sections[0]}**\n\n"  # Add the introduction
    output = ''
    for section in sections[1:]:
        if section.startswith("Case"):
            # Add case title
            output += f"### {section}\n\n"
        else:
            # Format the case details
            output += f"- {section.strip()}\n\n"

    return output

formatted_summary = format_case_summary(input_string)
print(formatted_summary)
