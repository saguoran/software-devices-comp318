import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"

prompt = """Back to Future: 👨👴🚗🕒
Batman: 🤵🦇
Transformers: 🚗🤖
Wonder Woman: 👸🏻👸🏼👸🏽👸🏾👸🏿
Winnie the Pooh: 🐻🐼🐻
The Godfather: 👨👩👧🕵🏻‍♂️👲💥
Game of Thrones: 🏹🗡🗡🏹
Spider-Man:"""


def openai_completion():
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.8,
        max_tokens=60,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )

    # print(response.choices[0].text)
    # movie name to icon
    print(f"{response.choices[0].text}")
    return f"{response.choices[0].text}"


if __name__ == '__main__':
    while True:
        movie_name = input("Movie Name: ")
        openai_completion(movie_name)
