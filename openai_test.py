import openai

# Author’s name, Last Modified by, date last Modified, Program description, Revision History

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = """Back to Future: 👨👴🚗🕒
Batman: 🤵🦇
Transformers: 🚗🤖
Wonder Woman: 👸🏻👸🏼👸🏽👸🏾👸🏿
Winnie the Pooh: 🐻🐼🐻
The Godfather: 👨👩👧🕵🏻‍♂️👲💥
Game of Thrones: 🏹🗡🗡🏹"""


def openai_completion():
    global prompt
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
    prompt += response.choices[0].text
    # print(response.choices[0].text)
    # movie name to icon
    # print(prompt)
    print(f"{response.choices[0].text}")
    return f"{response.choices[0].text}"


if __name__ == '__main__':
    while True:
        movie_name = input("Movie Name: ") + ":"
        prompt += "\n"
        prompt += movie_name
        openai_completion()
