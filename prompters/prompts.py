user_prompt = ("Analyze briefly and finally recommend next {category_singular} item I might purchase."
               "You should try to keep all your thoughts and responses as concise as possible and recommend next {category_singular} I might purchase inside {{emb_token}} and {{emb_end_token}} "
               "with estimation or guess about its all attributes (title, average_rating, rating_number and description). You don't need to utter irrelevant polite remarks or explanatory sentences."
               "For example: {{emb_token}}'title': {title}\n 'average_rating': {average_rating}\n 'rating_number': {rating_number}\n 'description': {description}{{emb_end_token}}")



##################### item #####################
item_prompt = "Summarize key attributes of the following {category_singular} inside {{emb_token}} and {{emb_end_token}}."


##################### categories ####################
cat_dict = {
    "Video_Games": ("video games", "video game", "NBA 2K17 - Early Tip Off Edition - PlayStation 4", "4.3", "223", "Following the record-breaking launch of NBA 2K16, the NBA 2K franchise continues to stake its claim as the most authentic sports video game with NBA 2K17. As the franchise that “all sports video games should aspire to be” (GamesRadar), NBA 2K17 will take the game to new heights and continue to blur the lines between video game and reality."),
    "CDs_and_Vinyl": ("CDs and vinyl", "CD or vinyl", "Release Some Tension", "4.6", "112", "['Swv ~ Release Some Tension']"),
    "Musical_Instruments": ("musical instruments", "musical instrument", "3 Mini Color Violin Fingering Tape for Fretboard Note Positions", "4.6", "840", "Finally, no more ugly masking tape!These plastic tapesmake it easier for students to learn fingering positions. This neat, durable tape can be easily and accurately placed on the fingerboard of any string instrument. Each package contains 1 roll of red, 1"),
    "Beauty": ("Beauty", "Beauty")
}


def obtain_prompts(category):
    """Obtain prompts for a specific category."""
    assert category in cat_dict, f"Category {category} not found in the dictionary."
    category, category_singular, title, average_rating, rating_number, description = cat_dict[category]
    return {
        "user_prompt": user_prompt.format(category=category, category_singular=category_singular, title=title, average_rating=average_rating, rating_number=rating_number, description=description),
        'item_prompt': item_prompt.format(category_singular=category_singular),
    }
