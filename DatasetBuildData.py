import random
from datasets import load_dataset

# ==========================================
# 1. THE PROMPT ENSEMBLE LEXICON
# ==========================================
# This provides the variation the Screwdriver needs to learn the latent *concept* # of the task, rather than memorizing a specific instruction string.

PROMPT_ENSEMBLES = {
    "neutral_baseline": [
        "The event occurred on a Tuesday afternoon.",
        "A person walked down the street.",
        "The object was placed on the table.",
        "It was a standard day in the city.",
        "The document contained several paragraphs."
    ],
    "sentiment": [
        "Analyze the sentiment of this text:",
        "Determine if the emotion is positive or negative:",
        "What is the attitude of the author?",
        "Evaluate the polarity of the sentence:",
        "Classify the feeling in this review:",
        "Is the tone of this passage optimistic or pessimistic?",
        "Assess the emotional valence of the following text:",
        "Identify the author's sentiment:"
    ],
    "topic_classification": [
        "Classify the news topic of this article:",
        "Determine the subject matter of this text:",
        "What is the primary theme of this passage?",
        "Categorize this news story:",
        "Identify the journalistic domain of this writing:",
        "Which news category does this text belong to?",
        "Determine the main topic discussed below:",
        "Assign a thematic label to this excerpt:"
    ],
    "nli_entailment": [
        "Determine the logical relationship between these statements:",
        "Does the premise entail, contradict, or remain neutral to the hypothesis?",
        "Analyze if the second sentence logically follows the first:",
        "Identify the textual entailment here:",
        "Are these two sentences contradictory?",
        "Assess the truth value of the hypothesis given the premise:",
        "Classify the inference between the premise and hypothesis:",
        "Evaluate the logical consistency of these two texts:"
    ],
    "qa_extraction": [
        "Extract the answer to the question from the context:",
        "Find the exact text span that answers this query:",
        "Read the passage and answer the following question:",
        "Locate the factual answer in the provided text:",
        "Identify the phrase that answers the user's question:",
        "Use the context to answer the prompt:",
        "Extract the target information from the paragraph:",
        "Find the solution to the question within this context:"
    ]
}

# ==========================================
# 2. RAW TEXT HARVESTERS
# ==========================================

def get_imdb_texts(sample_size=10000):
    print("      Loading IMDB (Sentiment)...")
    dataset = load_dataset("imdb", split="train")
    # Filter for reasonable lengths to save memory/compute during extraction
    short_reviews = [row['text'] for row in dataset if len(row['text'].split()) < 150]
    random.shuffle(short_reviews)
    return short_reviews[:sample_size]

def get_agnews_texts(sample_size=10000):
    print("      Loading AG News (Topic)...")
    dataset = load_dataset("ag_news", split="train")
    texts = [row['text'] for row in dataset]
    random.shuffle(texts)
    return texts[:sample_size]

def get_mnli_texts(sample_size=10000):
    print("      Loading MNLI (Entailment)...")
    dataset = load_dataset("glue", "mnli", split="train")
    # Stitch premise and hypothesis together for a single cohesive text block
    texts = [f"Premise: {row['premise']} Hypothesis: {row['hypothesis']}" for row in dataset]
    random.shuffle(texts)
    return texts[:sample_size]

def get_squad_texts(sample_size=10000):
    print("      Loading SQuAD (Question Answering)...")
    dataset = load_dataset("squad", split="train")
    # Stitch context and question together
    texts = [f"Context: {row['context']} Question: {row['question']}" for row in dataset]
    random.shuffle(texts)
    return texts[:sample_size]

# ==========================================
# 3. TASK POOL ASSEMBLER
# ==========================================

def build_master_task_pool():
    """
    Constructs the weighted dictionary for the massive data extraction loop.
    You can adjust the sample_size requests based on how many shards you want to build.
    """
    print("[*] Assembling Master Task Pool...")
    
    task_pool = {
        "imdb_sentiment": {
            "weight": 0.15, 
            "data": get_imdb_texts(15000), 
            "prompts": PROMPT_ENSEMBLES["sentiment"],
            "neutral": PROMPT_ENSEMBLES["neutral_baseline"]
        },
        "ag_news": {
            "weight": 0.15, 
            "data": get_agnews_texts(15000), 
            "prompts": PROMPT_ENSEMBLES["topic_classification"],
            "neutral": PROMPT_ENSEMBLES["neutral_baseline"]
        },
        "glue_mnli": {
            "weight": 0.35, 
            "data": get_mnli_texts(35000), 
            "prompts": PROMPT_ENSEMBLES["nli_entailment"],
            "neutral": PROMPT_ENSEMBLES["neutral_baseline"]
        },
        "squad_qa": {
            "weight": 0.35, 
            "data": get_squad_texts(35000), 
            "prompts": PROMPT_ENSEMBLES["qa_extraction"],
            "neutral": PROMPT_ENSEMBLES["neutral_baseline"]
        }
    }
    
    print("[+] Task Pool Ready.")
    return task_pool