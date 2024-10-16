import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob = dict()
    for j in corpus:
        if j not in prob:
            prob[j] = 0
        if corpus[page]:
            prob[j] += (1-damping_factor)/len(corpus)
        else:
            prob[j] += 1/len(corpus)
    if corpus[page]:
        for j in corpus[page]:
            prob[j] += damping_factor/len(corpus[page])
    return prob


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num = n
    sample = dict()
    key = random.choice(list(corpus.keys()))
    sample[key] = 1
    prob = transition_model(corpus, key, damping_factor)
    n -= 1
    while(n):
        key = random.choices(list(prob.keys()), weights=list(prob.values()), k=1)[0]
        if key not in sample:
            sample[key] = 1
        else:
            sample[key] += 1
        prob = transition_model(corpus, key, damping_factor)
        n -= 1
    print(sample)
    for i in sample:
        sample[i] = sample[i]/num
    return sample

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = dict()
    new_pagerank = dict()
    n = len(corpus)
    for i in corpus:
        if not corpus[i]:
            corpus[i] = set(corpus.keys())
        pagerank[i] = 1/n
        new_pagerank[i] = 0
    condition = True
    while(condition):
        for p in corpus:
            new_pagerank[p] = ((1-damping_factor)/n)+damping_factor*(link_page(corpus, p, pagerank))
        condition = loop_run(pagerank, new_pagerank)
        pagerank = new_pagerank.copy()
    return new_pagerank

def loop_run(pagerank, new_pagerank):
    c = 0
    for i in pagerank:
        if abs(pagerank[i] - new_pagerank[i]) <= 0.001:
            c += 1
    print(pagerank, new_pagerank)
    if c == len(pagerank):
        return False
    return True

def link_page(corpus, page, pagerank):
    rank = 0
    for i in corpus:
        if page in corpus[i]:
            rank += pagerank[i]/len(corpus[i])
    return rank

if __name__ == "__main__":
    main()