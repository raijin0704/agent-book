import argparse
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .agent import DocumentationAgent

# from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="ユーザー要求に基づいて要件定義書を作成します"
    )
    parser.add_argument(
        "--task", type=str, help="作成したいapplicationについて記載してください"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="生成するペルソナの人数を設定してください（デフォルト:5）",
    )
    args = parser.parse_args()

    llm = ChatOpenAI(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-1.5-flash",
        temperature=0,
    )
    agent = DocumentationAgent(llm=llm, k=args.k)
    final_output = agent.run(user_request=args.task)

    print(final_output)


if __name__ == "__main__":
    main()
