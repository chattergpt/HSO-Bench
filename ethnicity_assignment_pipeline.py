!pip install -q langchain langchain_google_genai google-generativeai langchain_community openai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.colab import userdata, drive
import re

# Select Model and Prompt Mode
model_choice = "gemini"  # options: "gemini", "openai"
prompt_mode = "vanilla_spec"  # options: "vanilla", "cot", "rule-guided"
scale_type = f"{model_choice}_{prompt_mode}"

GOOGLE_API_KEY = userdata.get("GeminiKey")
OPENAI_API_KEY = userdata.get("OpenAIKey")

if model_choice == "gemini":
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0
    )
elif model_choice == "openai":
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # or "gpt-3.5-turbo"
        openai_api_key=OPENAI_API_KEY,
        temperature=0.0
    )
else:
    raise ValueError("Invalid model choice. Use 'gemini' or 'openai'.")

def get_prompt_template(mode="vanilla"):
    preamble = """
You are a knowledgeable cultural sociologist.

I am a person in {country}, and here is what I identify as: {identity}.

Using these numerical categories as a scale, give a number and explanation that best describes my situation. Your response must be based on the given definitions. Do not infer additional factors outside the definitions provided.

Oppression Scale Definitions:
1 - Little to No Oppression: These groups constitute the most privileged in their society. They have not experienced systemic oppression or violence on the basis of their identity, and they experience no social risk factors. They were historically aligned with or integrated into dominant societal institutions, laws, or culture of their society and were able to easily assimiate into it. Over time, they have maintained extensive access to societal resources, opportunities, and benefits and enjoy the most institutional privilege. They are highly unlikely to encounter systemic barriers, discrimination, violence, or other oppression based on their race or ethnicity, and they do not encounter bias, stereotyping, or tokenism on the basis of their ethnic and/or racial identity.
2 - Low Oppression: These groups constitute the second most privileged in their society. They have experienced minimal systemic oppression or violence on the basis of their identity, and have minimal social risk factors. They are able to assimilate with relatively few obstacles into the dominant institutions, laws, or culture of their society. Over time, they have maintained broad access to societal resources, opportunities, and benefits and enjoy institutional privilege with some exceptions. They are moderately unlikely to encounter systemic barriers, discrimination, violence, or other oppression based on their race or ethnicity, but they may encounter occasional bias, stereotyping, or tokenism on the basis of their ethnic and/or racial identity.
3 - Moderate Oppression: These groups constitute the middle group in terms of privilege versus oppression in their society. They have experienced moderate systemic oppression or violence on the basis of their identity, and have moderate social risk factors. They are not easily able to assimilate into the dominant institutions, laws, or culture of their society. Over time, they have maintained some access to societal resources, opportunities, and benefits and enjoy some institutional privileges. They have maintained some but incomplete access to institutional power, and face recurring bias, stereotyping, or discrimination. They are often visible in some domains but underrepresented or stereotyped in others.
4 - High Oppression: These groups have experienced longstanding, institutionalized exclusion, including colonization, forced assimilation, segregation, or legal discrimination. While they may have been partially included in legal or social systems, they were treated as subordinate populations with restricted rights, power, or recognition. Today, they continue to face persistent systemic disadvantage and under-representation, with limited progress in inclusion or representation.
5 - Severe Oppression: These groups constitute the most oppressed group in their society. They have experienced severe levels of systemic oppression or violence on the basis of their identity, and have severe social risk factors. They are actively excluded from the dominant institutions, laws, or culture of their society. Over time, they have experienced little access to societal resources, opportunities, and benefits and do not experience almost any institutional privilege. They are the most likely to encounter systemic barriers, discrimination, violence, or other oppression based on their race or ethnicity, and they very frequently encounter bias, stereotyping, or tokenism on the basis of their ethnic and/or racial identity.
"""

    rules = """
Follow these rules when assigning a category:
1. This classification must be based solely on historical and systemic factors of oppression. Do not consider cultural contributions, economic success, or individual achievements when assigning a category.
2. Do not assume that globally marginalized identities (e.g., Asian, Jewish, Latino) experience systemic oppression in the same way across all societies. Your classification must be based strictly on the historical and systemic role of that identity group within {country}.
3. When someone identifies using a national label (e.g., “Canadian,” “Brazilian,” “American”), assume they are referring to the dominant racial or ethnic group in that country, unless the label includes an additional modifier that indicates a minority or marginalized population.
4. When someone identifies as having both privileged and historically oppressed ancestries, lean toward the rating that reflects the marginalized component, especially if the society has historically assigned group membership or social treatment based on that marginalized identity. Many societies treat such individuals as non-members of the dominant group, regardless of partial privileged heritage.
5. If no evidence exists of structural disadvantage, assign a low score. Do not infer oppression based on general trends, recent events, or social stereotypes not grounded in the long-term history of systemic oppression in {country}.
6. Only assign a 4 or higher if the group has faced long-term, institutionalized exclusion across multiple major domains (e.g., housing, education, voting, etc.), with limited inclusion efforts over time. Consider category 3 if the group has experienced discrimination, stereotyping, or underrepresentation, but has maintained meaningful access to education, employment, and civic institutions.
"""

    step_by_step = """
Think step by step before coming to your final answer.

Start by answering these three questions:
1. Has this group historically faced institutional violence, exclusion, or systemic oppression within {country}?
2. How were they positioned within dominant institutions (government, education, legal system) of {country}?
3. Did they experience long-term structural disadvantage compared to dominant groups in {country}?

After answering each question, decide which oppression scale level (1–5) fits best.
"""

    format_answer = """
Format your final answer like this:

If the identity label does not clearly match an ethnic or racial group, respond like this:
Rating: None
Explanation: <brief explanation for why no score is given>

Otherwise, respond like this:
Rating: <number from 1 to 5>
Explanation: <brief explanation based on the context>
"""

    if mode == "vanilla":
        prompt = preamble + format_answer
    elif mode == "cot":
        prompt = preamble + step_by_step + format_answer
    elif mode == "rule-guided":
        prompt = preamble + rules + format_answer
    else:
        raise ValueError("Unknown prompt mode")

    return PromptTemplate(input_variables=["identity", "country"], template=prompt)

# Build Chain
prompt = get_prompt_template(prompt_mode)
chain = LLMChain(llm=llm, prompt=prompt)

drive.mount('/content/drive')
excel_path = "/content/drive/My Drive/Dye Lab/unmatched_identities.xlsx"
all_sheets = pd.read_excel(excel_path, sheet_name=None)

# Process Rows
def process_row(row, sheet_name):
    identity = row["identity"]
    country = row["country"]
    try:
        response = chain.run(identity=identity, country=country)
        match_rating = re.search(r"Rating:\s*([1-5])", response)
        match_expl = re.search(r"Explanation:\s*(.*)", response, re.DOTALL)
        rating = int(match_rating.group(1)) if match_rating else None
        explanation = match_expl.group(1).strip() if match_expl else response
    except Exception as e:
        rating, explanation = None, f"ERROR: {e}"

    return {
        "identity": identity,
        "country": country,
        "sheet_name": sheet_name,
        f"{scale_type}_rating": rating,
        f"{scale_type}_explanation": explanation
    }

all_results = []
for sheet_name, df in all_sheets.items():
    print(f"Processing sheet: {sheet_name}")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_row, row, sheet_name) for _, row in df.iterrows()]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Sheet: {sheet_name}"):
            all_results.append(f.result())

results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{scale_type}.csv", index=False)
