Spacy Models for Biomedical Text.

| Feature | Description |
| --- | --- |
| **Name** | `en_ner_bionlp13cg_md` |
| **Version** | `0.5.1` |
| **spaCy** | `>=3.4.1,<3.5.0` |
| **Default Pipeline** | `tok2vec`, `tagger`, `attribute_ruler`, `lemmatizer`, `parser`, `ner` |
| **Components** | `tok2vec`, `tagger`, `attribute_ruler`, `lemmatizer`, `parser`, `ner` |
| **Vectors** | 4087446 keys, 50000 unique vectors (200 dimensions) |
| **Sources** | BIONLP13CG<br />OntoNotes 5<br />Common Crawl<br />GENIA 1.0 |
| **License** | `CC BY-SA 3.0` |
| **Author** | [Allen Institute for Artificial Intelligence](https://allenai.github.io/SciSpaCy/) |

### Label Scheme

<details>

<summary>View label scheme (113 labels for 3 components)</summary>

| Component | Labels |
| --- | --- |
| **`tagger`** | `$`, `''`, `,`, `-LRB-`, `-RRB-`, `.`, `:`, `ADD`, `AFX`, `CC`, `CD`, `DT`, `EX`, `FW`, `HYPH`, `IN`, `JJ`, `JJR`, `JJS`, `LS`, `MD`, `NFP`, `NN`, `NNP`, `NNPS`, `NNS`, `PDT`, `POS`, `PRP`, `PRP$`, `RB`, `RBR`, `RBS`, `RP`, `SYM`, `TO`, `UH`, `VB`, `VBD`, `VBG`, `VBN`, `VBP`, `VBZ`, `WDT`, `WP`, `WP$`, `WRB`, `XX`, ```` |
| **`parser`** | `ROOT`, `acl`, `acl:relcl`, `acomp`, `advcl`, `advmod`, `amod`, `amod@nmod`, `appos`, `attr`, `aux`, `auxpass`, `case`, `cc`, `cc:preconj`, `ccomp`, `compound`, `compound:prt`, `conj`, `cop`, `csubj`, `dative`, `dep`, `det`, `det:predet`, `dobj`, `expl`, `intj`, `mark`, `meta`, `mwe`, `neg`, `nmod`, `nmod:npmod`, `nmod:poss`, `nmod:tmod`, `nsubj`, `nsubjpass`, `nummod`, `parataxis`, `pcomp`, `pobj`, `preconj`, `predet`, `prep`, `punct`, `quantmod`, `xcomp` |
| **`ner`** | `AMINO_ACID`, `ANATOMICAL_SYSTEM`, `CANCER`, `CELL`, `CELLULAR_COMPONENT`, `DEVELOPING_ANATOMICAL_STRUCTURE`, `GENE_OR_GENE_PRODUCT`, `IMMATERIAL_ANATOMICAL_ENTITY`, `MULTI_TISSUE_STRUCTURE`, `ORGAN`, `ORGANISM`, `ORGANISM_SUBDIVISION`, `ORGANISM_SUBSTANCE`, `PATHOLOGICAL_FORMATION`, `SIMPLE_CHEMICAL`, `TISSUE` |

</details>

### Accuracy

| Type | Score |
| --- | --- |
| `TAG_ACC` | 0.00 |
| `LEMMA_ACC` | 0.00 |
| `DEP_UAS` | 0.00 |
| `DEP_LAS` | 0.00 |
| `DEP_LAS_PER_TYPE` | 0.00 |
| `SENTS_P` | 0.00 |
| `SENTS_R` | 0.00 |
| `SENTS_F` | 0.00 |
| `ENTS_F` | 77.02 |
| `ENTS_P` | 79.64 |
| `ENTS_R` | 74.57 |
| `NER_LOSS` | 511815.13 |