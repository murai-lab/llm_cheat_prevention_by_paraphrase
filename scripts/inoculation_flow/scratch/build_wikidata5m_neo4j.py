# Jeremy Lim
# jlim@wpi.edu

# https://deepgraphlearning.github.io/project/wikidata5m
# quick testing script for ingesting wikidata5m, and building a local database using neo4j
# adapting parts of neo4j example...

# https://neo4j.com/docs/getting-started/languages-guides/neo4j-python/

# https://pypi.org/project/neo4j/
# https://neo4j.com/docs/python-manual/current/
# Query cheat sheet: https://neo4j.com/docs/cypher-cheat-sheet/5/auradb-enterprise
# https://neo4j.com/docs/getting-started/appendix/graphdb-concepts/#graphdb-relationship-type

# Might look at this later: https://neo4j.com/docs/cypher-manual/current/clauses/load-csv/

# Will create in batches for speed: https://community.neo4j.com/t/batch-create-and-merge-statements/20691/4


import sys, os
import math, random
import pickle
import datetime

from neo4j import GraphDatabase
import bert_score
from matplotlib import pyplot as plt
import numpy as np


CORPUS_CHAR_LIMIT = 135  # Rough character limit that should encompass enough context for BERTScore to select a proper alias.
# Try 150 instead of 200...
# NOTE: BERTScore cannot go longer than 510 characters, due to model size limit!
MAX_COMPARE_CHOICES = 50  # We can only compare so many options at once, computationally.

def build_first_ent_rel_table(fpath):
    """
    Import file, but only include the first relation/entity for every marker
    """
    mapping_table = {}

    # Dictionary that just lists the order we ingested data. Use to easily select random nodes.
    id_table = {}
    idx = 0

    with open(fpath, "r") as f:
        for fline in f:
            # split by tabs
            linesplit = fline.replace("\n", "").split("\t")

            mapping_table[linesplit[0]] = linesplit[1]
            id_table[linesplit[0]] = idx
            idx += 1

    return mapping_table, id_table, idx

# functions provided to make names in the graph somewhat match neo4j's standard...
def to_underscore(in_str):
    return in_str.replace(" ", "_")


def from_underscore(in_str):
    return in_str.replace("_", " ")


def alias_stats(fpath):

    alias_counts = []
    # number_evals = 0

    with open(fpath, "r") as f:
        for fline in f:
            # split by tabs
            linesplit = fline.replace("\n", "").split("\t")

            alias_counts.append(len(linesplit))
            # number_evals += math.factorial(len(linesplit))

            if len(linesplit) >= 30:
                linesort = sorted(linesplit, key=lambda x: len(x))
                print("break")

    print("Min # aliases: " + str(min(alias_counts)))
    print("Average # aliases: " + str(np.mean(alias_counts)))
    print("Max # aliases: " + str(max(alias_counts)))

    # print("Number of evaluations (naive): " + str(number_evals))

    plt.title("Alias Distribution")
    plt.hist(alias_counts)
    plt.show()

def space_token_matches(reference_text, alias_list):
    # Return a list of aliases that have a space-separated token that exists in the reference text.

    new_aliases = []

    for candidate in alias_list:
        candidate_tokens = candidate.split(" ")
        for token in candidate_tokens:
            if token.lower() in reference_text:
                new_aliases.append(candidate)
                break

    if len(new_aliases) != 0:
        return new_aliases
    else:
        return alias_list


# Bertscore notes:
# Understanding why they use the terms "precision" and "recall" for the bertscore evaluations: https://en.wikipedia.org/wiki/Precision_and_recall
# BERTScore Recall is analagous to sensitivity; how well can you find tokens in candidate matching the reference? Reference is considered the ground truth in a way.
# BERTScore Precision

def filter_ent_aliases(ent_fpath, corpus_fpath):
    # Filter entities on the first sentence of the corpus
    #
    # First: load some text for each entity from corpus.

    # Bert score, use distilbert for now.
    scorer = bert_score.BERTScorer(model_type="distilbert-base-uncased", lang="en")

    print("Load corpus")
    corpus_dict = {}
    with open(corpus_fpath, "r") as corpus_file:
        for fline in corpus_file:
            linesplit = fline.find("\t")
            entity = fline[:linesplit]
            text = fline[linesplit+1:CORPUS_CHAR_LIMIT+linesplit+1]
            corpus_dict[entity] = text

    ent_mapping_table = {}

    # Dictionary that just lists the order we ingested data. Use to easily select random nodes.
    id_table = {}
    idx = 0

    missing_corpus_count = 0

    print("Load/filter entities:")
    # find total number of lines first? Basic.
    with open(ent_fpath, "r") as f:
        line_counter = 0
        for _ in f:
            line_counter += 1

    print("Totl entity line count: " + str(line_counter))

    start = datetime.datetime.now()

    # Load/process entity
    with open(ent_fpath, "r") as f:
        for line_count, fline in enumerate(f):
            if line_count % 1000 == 0:
                print("Lines read: " + str(line_count) + "/" + str(line_counter))

                if line_count != 0:
                    elapsed = (datetime.datetime.now() - start)
                    remaining = line_counter - line_count

                    time_per_line = elapsed / line_count

                    estimated_remaining_seconds = remaining * time_per_line

                    print("Estimated time remaining: " + str(estimated_remaining_seconds))

            # split by tabs
            linesplit = fline.replace("\n", "").split("\t")
            q_identifier = linesplit[0]
            alias_choices = linesplit[1:]

            # Dedupe the choices
            alias_choices = dedupe_string_list(alias_choices)

            if len(alias_choices) == 1:
                ent_mapping_table[q_identifier] = alias_choices[0]
            else:
                # print("# alias choices: " + str(len(alias_choices)))

                # if len(alias_choices) > 1000:
                #     print("Break")

                # Using BertScore on the first part of the corpus to rank aliases...
                # Use precision, with aliases as candidates. The aliases will probably be much shorter, so
                # Better to match candidates to longer reference text only.

                try:
                    reference_text = corpus_dict[q_identifier]
                except KeyError as e:
                    # print("SKIP: Missing corpus text for this entity!")
                    # keep track of how many
                    missing_corpus_count += 1
                    continue

                # remove choices that are absent from corpus
                orig_alias_choices = alias_choices

                alias_choices = [x for x in alias_choices if x.lower() in reference_text.lower()]

                if len(alias_choices) == 0:  # No string matching, fall back to bertscore.

                    # print("break")
                    # if you can't do simple string matching... use bertscore then
                    alias_choices = orig_alias_choices

                    # Filter more by looking at individual word matching!
                    # But will return the full list if there's no matches at all as a fallback.
                    alias_choices = space_token_matches(reference_text.lower(), alias_choices)

                    if len(alias_choices) == 1:
                        ent_mapping_table[q_identifier] = alias_choices[0]
                        continue

                    # Build a set of references/candidates to evaluate.
                    ref_list = []
                    candidate_list = []
                    for idx in range(len(alias_choices)):
                        ref_list.append(reference_text)
                        candidate_list.append(alias_choices[idx])

                    # use P
                    P, R, F1 = scorer.score(candidate_list, ref_list)

                    # Take the highest precision. The aliases are likely to be much smaller, so we want to match them into
                    # the larger reference text.
                    best_idx = np.argmax(np.array(P))
                    # print("Ref text: " + reference_text)
                    # print("Alias list: " + str(alias_choices))
                    # print("Best candidate: " + alias_choices[best_idx])

                    ent_mapping_table[q_identifier] = alias_choices[best_idx]
                elif len(alias_choices) == 1:
                    ent_mapping_table[q_identifier] = alias_choices[0]
                else:
                    # multiple options, likely all valid. Choose one at random.
                    ent_mapping_table[q_identifier] = random.choice(alias_choices)


            id_table[q_identifier] = idx
            idx += 1

    print("Entities with missing corpus that were skipped: " + str(missing_corpus_count))

    return ent_mapping_table, id_table, idx


def dedupe_string_list(str_list):
    # Remove exact duplicates. Also, aliases that differ only in capitalization, favoring more capital letters than less.
    str_mapping = {}
    best_upper_counts = {}

    # count uppercase inspiration: https://stackoverflow.com/questions/18129830/count-the-uppercase-letters-in-a-string-with-python
    for item in str_list:
        uppercount = sum(1 for x in item if x.upper())
        lowered = item.lower()
        if lowered not in str_mapping:
            str_mapping[lowered] = item
            best_upper_counts[lowered] = uppercount
        else:
            # Check the upper count.
            if uppercount > best_upper_counts[lowered]:
                best_upper_counts[lowered] = uppercount
                str_mapping[lowered] = item

    out_list = []
    # rebuild list now. Original order not guaranteed!
    for key in str_mapping.keys():
        out_list.append(str_mapping[key])

    return out_list
# Bertscore examples:
# https://medium.com/@abonia/bertscore-explained-in-5-minutes-0b98553bfb71
# https://huggingface.co/spaces/evaluate-metric/bertscore
def filter_relation_aliases(rel_fpath):

    # Bert score, use distilbert for now.
    scorer = bert_score.BERTScorer(model_type="distilbert-base-uncased", lang="en")

    rel_mapping_table = {}

    # Dictionary that just lists the order we ingested data. Use to easily select random nodes.
    id_table = {}
    idx = 0

    # Load/process entity
    with open(rel_fpath, "r") as f:
        for fline in f:
            # split by tabs
            linesplit = fline.replace("\n", "").split("\t")
            p_identifier = linesplit[0]
            alias_choices = linesplit[1:]

            # Dedupe the choices
            alias_choices = dedupe_string_list(alias_choices)

            if len(alias_choices) == 1:
                rel_mapping_table[p_identifier] = alias_choices[0]  # TODO: Use Corpus/Bertscore to rank relations pairwise.
            else:
                # print("# alias choices: " + str(len(alias_choices)))
                # Which alias is most similar to all other aliases?
                # Run Bertscore Recall vs all other tokens, using current one as reference.
                # Rank these by average recall...

                if len(alias_choices) > MAX_COMPARE_CHOICES:
                    # sub-select randomly? No better/worse than always just blindly selecting the first alias.
                    alias_choices = random.choices(alias_choices, k=MAX_COMPARE_CHOICES)

                # Build a set of references/candidates to evaluate.
                ref_list = []
                candidate_list = []
                for idx in range(len(alias_choices)):
                    reference = alias_choices[idx]
                    remaining_choices = alias_choices[:idx] + alias_choices[(idx+1):]
                    for candidate in remaining_choices:
                        ref_list.append(reference)
                        candidate_list.append(candidate)

                # use R
                P, R, F1 = scorer.score(candidate_list, ref_list)

                # reduce, average.
                reduce_R = np.array(R).reshape(-1, (len(alias_choices) - 1))
                reduce_R = np.mean(reduce_R, axis=1, keepdims=False)

                # choose the highest average one.
                best_idx = np.argmax(reduce_R)
                # print("Alias list: " + str(alias_choices))
                # print("Best candidate: " + alias_choices[best_idx])

                rel_mapping_table[p_identifier] = alias_choices[best_idx]


            id_table[p_identifier] = idx
            idx += 1

    return rel_mapping_table

def import_train_to_db():
    print("Start")
    # This works, but is quite slow...
    # NOTE: LLM for choosing more common mappings then the first one listed?

    batch_size = 5000

    # Import and putting in entities/relations directly into database.
    root_path = "/home/jeremy/Documents/WPI_Summer_24/glaze_adv_llm/Project/Wikidata5M"

    relation_path = os.path.join(root_path, "wikidata5m_relation.txt")
    entity_path = os.path.join(root_path, "wikidata5m_entity.txt")
    text_corpus_path = os.path.join(root_path, "wikidata5m_text.txt")


    # print("relation alias distribution: ")
    # alias_stats(relation_path)
    # print("Entity alias distribution: ")
    # alias_stats(entity_path)

    print("Filtering relation aliases...")
    if os.path.exists("relation.pickle"):
        with open("relation.pickle", "rb") as f:
            relation_mapping = pickle.load(f)
    else:
        # relation_mapping, notused, nope = build_first_ent_rel_table(relation_path)
        relation_mapping = filter_relation_aliases(relation_path)

        # Dump relations so we don't have to re-calculate constantly.
        with open("relation.pickle", "wb") as f:
            pickle.dump(relation_mapping, f)



    print("Filtering entity aliases...")

    if os.path.exists("entity.pickle"):
        with open("entity.pickle", "rb") as f:
            entity_dict = pickle.load(f)
            entity_mapping = entity_dict['entity_mapping']
            entity_add_ids = entity_dict['entity_add_ids']
            highest_idx = entity_dict['highest_idx']
    else:
        entity_mapping, entity_add_ids, highest_idx =filter_ent_aliases(entity_path, text_corpus_path)
        # entity_mapping, entity_add_ids, highest_idx = build_first_ent_rel_table(entity_path)

        # Dump output so we don't redo it many, many times...
        with open("entity.pickle", "wb") as f:
            entity_object = {
                "entity_mapping": entity_mapping,
                "entity_add_ids": entity_add_ids,
                "highest_idx": highest_idx,
            }
            pickle.dump(entity_object, f)


    # return  # Early return for now...
    train_graph_path = os.path.join(root_path, "wikidata5m_inductive", "wikidata5m_inductive_train.txt")

    print("Highest entity idx: " + str(highest_idx))

    uri = "neo4j://localhost:7687"
    # Really basic local database for development. Not a production instance...
    auth = ("neo4j", "abcd1234")

    database_name = "neo4j"

    # The example code provides some insight into using MERGE:
    #     driver.execute_query(
    #         "MERGE (a:Person {name: $name}) "
    #         "MERGE (friend:Person {name: $friend_name}) "
    #         "MERGE (a)-[:KNOWS]->(friend)",
    #         name=name, friend_name=friend_name, database_="neo4j",
    #     )
    # Will create, or ignore if exists.

    line_count = 0

    skip_rel = 0


    # Will encode most of the information in the "name" property. All nodes are Object, all relations are RELATION
    with GraphDatabase.driver(uri, auth=auth) as driver:

        driver.execute_query(
            "CREATE INDEX obj_index IF NOT EXISTS FOR (n:Object) on (n.name)",
            database_=database_name
        )

        # NOTE! Start with empty database...
        # print("Creating all nodes...")
        #
        # # Using a session, to batch many commands at once...
        # with driver.session() as session:
        #
        #     # Let's try adding an index to the object names, to speed up performance.
        #     driver.execute_query(
        #         "CREATE INDEX obj_index IF NOT EXISTS FOR (n:Object) on (n.name)",
        #         database_=database_name
        #     )
        #
        #     tx = session.begin_transaction()
        #
        #     for idx, k in enumerate(entity_mapping.keys()):
        #         if idx % batch_size == 0:
        #             tx.commit()
        #             print("Entities added to DB: " + str(idx))
        #             tx = session.begin_transaction()
        #
        #         name = entity_mapping[k]
        #         tx.run(
        #             "CREATE (a:Object {name: $namea})",
        #             namea=name, database_=database_name
        #         )
        #
        #     if not tx.closed():
        #         tx.commit()

        # Only incremented when an entity is used in some relationship!
        entity_idx = 0
        used_entity_table = {}

        print("Creating all relationships...")
        # also, load the graph file.
        with open(train_graph_path, "r") as f:

            with driver.session() as session:
                tx = session.begin_transaction()

                for idx, fline in enumerate(f):
                    if idx % batch_size == 0:
                        tx.commit()
                        print("Relationships added to DB: " + str(idx))
                        print("Next addidx: " + str(entity_idx))
                        tx = session.begin_transaction()

                    file_line = fline.replace("\n", "").split("\t")

                    try:
                        enta = entity_mapping[file_line[0]]
                        entb = entity_mapping[file_line[2]]
                        relation = relation_mapping[file_line[1]]
                        # Do these first; make sure they exist!
                    except KeyError as e:
                        # apparently, some keys are still missing!
                        # If the key is missing, skip that relationship.
                        skip_rel += 1
                        continue

                    # Id logic
                    if not enta in used_entity_table:
                        used_entity_table[enta] = entity_idx
                        entity_idx += 1
                    ida = used_entity_table[enta]

                    # Id logic
                    if not entb in used_entity_table:
                        used_entity_table[entb] = entity_idx
                        entity_idx += 1
                    idb = used_entity_table[entb]

                    tx.run(
                        "MERGE (a:Object {name: $namea, addidx: $ida})\
                         MERGE (b:Object {name: $nameb, addidx: $idb})\
                         MERGE (a)-[:RELATION {name: $relname}]->(b)",
                        namea=enta, ida=ida, nameb=entb, idb=idb, relname=relation, database_=database_name
                    )

                    # tx.run(
                    #     "CREATE (a:Object {name: $namea})-[:RELATION {name: $relname}]->(b:Object {name: $nameb})",
                    #     namea=enta, nameb=entb, relname=relation, database_=database_name
                    # )

                    # Switch to create, based queries.

                    # Improved query; let neo4j optimize better?
                    # driver.execute_query(
                    #     "MERGE (a:Object {name: $namea})-[:RELATION {name: $relname}]->(b:Object {name: $nameb})",
                    #     namea=enta, nameb=entb, relname=relation, database_=database_name
                    # )

                    line_count += 1
                    # if line_count % 1000 == 0:
                    #     print("relationships added to DB: " + str(line_count))

                # Make sure to commit leftovers!
                if not tx.closed():
                    tx.commit()

    print("Max entity idx: " + str(entity_idx))
    print("Relationships added: " + str(line_count))
    print("Relationships skipped: " + str(skip_rel))
    print("Done")


if __name__ == "__main__":
    import_train_to_db()


# Notes on selecting random nodes:
# https://stackoverflow.com/questions/23994830/selecting-random-nodes-neo4j
# Random seeds not supported: https://stackoverflow.com/questions/73096984/fix-random-seed-in-neo4j
# Added a specific property that just enumerates each node by add order, called "addidx".
# max value: 4813491
# uh-oh, not all of these entities are used!
# Adjusted logic to do ids for only entities that were used, so we don't have a case where we randomly select non-existent nodes...
# Other notes:
# Match and filter by degree: https://stackoverflow.com/questions/22346526/how-to-count-the-number-of-relationships-in-neo4j
# Advanced query: https://community.neo4j.com/t/cypher-query-to-count-the-number-of-relationships-of-specific-type-that-each-node-has-including-same-type-relationships-in-their-sub-nodes-ask-question/17987/2
# Simple way to select random node: "match (n) return n order by rand() limit 1"
# Querying by degree: https://stackoverflow.com/questions/32775427/how-to-get-degree-of-a-node-while-filtering-using-match-in-neo4j
# really good query example: https://stackoverflow.com/questions/56918811/neo4j-is-it-possible-and-how-group-all-nodes-with-matching-property-values-by-th
# Next addidx: 4574679