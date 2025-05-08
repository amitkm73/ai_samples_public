"""
text embeddings and semantic distances
"""
import sys
import os
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import seaborn as sns

SYS_SUCCESS = 0
SYS_FAILURE = 1

def read_sentences(file_path):
    """ assumes file exists """
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]


def get_embeddings(client, sentences):
    """ compute embeddings for a list of sentences using OpenAI's API. """
    embeddings = []
    for sentence in sentences:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=sentence
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)


def plot_distance_matrix(sentences, distances):
    """ heatmap of the distance matrix """
    # short labels for the sentences
    short_labels = []
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        short_label = f"{i+1}. {' '.join(words[:3])}..."
        short_labels.append(short_label)
    plt.figure(figsize=(12, 10))
    plt.gcf().canvas.manager.set_window_title('sentence distances heatmap')
    # heatmap using seaborn
    sns.heatmap(
        distances,
        annot=True,  # distance values
        fmt='.3f',   # 3 decimal places
        cmap='YlGnBu_r',  # color map (reversed so darker = more similar)
        xticklabels=short_labels,
        yticklabels=short_labels,
        square=True  # cells square
    )
    # rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    # plt.title('cosine distances between sentences', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('sentence_distances_heatmap.png', dpi=300, bbox_inches='tight')
    print("heatmap saved as 'sentence_distances_heatmap.png'")
    plt.show()


def main():
    """ main function to process sentences and compute similarity """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(SYS_FAILURE)
    client = OpenAI(api_key=api_key)
    sentences = read_sentences("sentences.txt")
    print(f"loaded {len(sentences)} sentences")
    print("computing embeddings...")
    embeddings = get_embeddings(client, sentences)
    distances = cosine_distances(embeddings)
    print("\ncosine distances between sentences:")
    print("-" * 50)
    for i, sent1 in enumerate(sentences):
        for j, sent2 in enumerate(sentences[i + 1:], start=i + 1):
            distance = distances[i, j]
            sent1_short = (sent1[:40] + '...') if len(sent1) > 40 else sent1
            sent2_short = (sent2[:40] + '...') if len(sent2) > 40 else sent2
            print(f"distance between \"{sent1_short}\" and \"{sent2_short}\" : {distance:.4f}")
    print("\ngenerating distance matrix visualization...")
    plot_distance_matrix(sentences, distances)
    sys.exit(SYS_SUCCESS)

if __name__ == "__main__":
    main()
