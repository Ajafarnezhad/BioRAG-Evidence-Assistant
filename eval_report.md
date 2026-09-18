# RAG Chunking Evaluation Report

This report summarizes the performance of 4 different chunking strategies on the 50-question evaluation dataset.

## Summary Metrics Table

| Strategy | QA Accuracy | Retrieval Coverage | P50 Retrieval Latency | P95 Retrieval Latency | P50 Answer Latency | P95 Answer Latency | Avg Cost |

| --- | --- | --- | --- | --- | --- | --- | --- |

| **fixed** | 18.0% | 58.0% | 11.7ms | 19.4ms | 90.4ms | 15387.9ms | $0.000000 |

| **sentence** | 12.0% | 60.0% | 12.2ms | 13.4ms | 384.0ms | 11950.9ms | $0.000000 |

| **paragraph** | 16.0% | 66.0% | 12.2ms | 14.7ms | 80.3ms | 12655.5ms | $0.000000 |

| **recursive** | 18.0% | 68.0% | 12.0ms | 14.3ms | 99.5ms | 13414.5ms | $0.000000 |


## Evidence of Chunking-Induced Hallucinations

Below are specific cases showing how poor chunking (especially `fixed`) directly causes retrieval failure or context fragmentation, forcing the LLM to hallucinate or fail.

### Case 1: How many active clients does Infosys have?

**Expected Answer:** `1,869`


#### Fixed-Size Chunking (Failure / Hallucination)

- **Retrieved Chunks Contain Expected Answer:** False

- **LLM Generated Answer:**
```
Answer:
I don't have enough information in the provided documents.

Sources Used:

Confidence:
Low
```

- **Top Retrieved Chunks:**

  * **Chunk 1:** `... Infosys Integrated Annual Report 2024-25288ParticularsYear ended March 31, 
2025 2024
Stater N.V. 3 –
Stater Nederland B.V. 69 74 
 2,580 2,883 
Sale of shared services including facilities 
and personnel
EdgeVerve Systems Limited 47 25 
Panaya Ltd 10 8 
GuideVision, s.r.o. 5 –
Infy Consulting Compa ...`

  * **Chunk 2:** `... Infosys Integrated Annual Report 2024-25272Particulars Year ended March 31,
2025 2024
Cost of software packages and others 
For own use 1,947 1,635 
Third-party items bought for 
service delivery to clients 7,670 5,256 
 9,617 6,891 
Other expenses 
Power and fuel 196 172 
Brand and marketing 1,067  ...`


#### Recursive Chunking (Success)

- **Retrieved Chunks Contain Expected Answer:** True

- **LLM Generated Answer:**
```
Answer:
Infosys has **1,869 active clients**.

Sources Used:
[5]

Confidence:
High
```

- **Top Retrieved Chunks:**

  * **Chunk 1:** `... Infosys Integrated Annual Report 2024-25288ParticularsYear ended March 31, 
2025 2024
Stater N.V. 3 –
Stater Nederland B.V. 69 74 
 2,580 2,883 
Sale of shared services including facilities 
and personnel
EdgeVerve Systems Limited 47 25 
Panaya Ltd 10 8 
GuideVision, s.r.o. 5 –
Infy Consulting Compa ...`

  * **Chunk 2:** `... Infosys Integrated Annual Report 2024-2581Annexure 3 – Particulars of employees
We are a leading provider of consulting, technology, outsourcing, and next-generation digital services. We enable clients across 59 countries to outperform their competition 
and stay ahead on the innovation curve. The r ...`


---

### Case 2: What were total revenues in fiscal 2025?

**Expected Answer:** `1,62,990 cr`


#### Fixed-Size Chunking (Failure / Hallucination)

- **Retrieved Chunks Contain Expected Answer:** False

- **LLM Generated Answer:**
```
Answer:
₹5,669 crore

Sources Used:
[Source 4]

Confidence:
High
```

- **Top Retrieved Chunks:**

  * **Chunk 1:** `...  fiscal 2025.
Composition and attendance
100%
Attendance100%
Independence3
Members4
MeetingsCorporat e Social Responsibility Committee ...`

  * **Chunk 2:** `... 492 crore for fiscal 2025 and 
2024, respectively. On consolidated basis is ₹585 crore and ₹533 
cro re for fiscal 2025 and 2024, respectively.
3. Other income and finance cost
Other income primarily includes income from investments, gain 
/ loss on investments, foreign exchange gain / loss on forwa ...`


#### Recursive Chunking (Success)

- **Retrieved Chunks Contain Expected Answer:** False

- **LLM Generated Answer:**
```
Answer:
₹1,62,990 crore

Sources Used:
[Source 2]

Confidence:
High
```

- **Top Retrieved Chunks:**

  * **Chunk 1:** `... ount of ongoing projects of fiscals 
2022, 2023 and 2024. The CSR amount spent in the US, Australia, and various parts of Europe, is over and above the statutory 
requirement in India. During fiscal 2025, the aggregate amount 
spent outside India was ` 40 crore.
Sd/-
Mumbai Govind Iyer
ChairpersonAp ...`

  * **Chunk 2:** `... in revenue, excluding the impact of currency 
fluctuations. We calculate constant currency growth by comparing current-period revenues in respective local currencies 
converted to USD using prior-period exchange rates and 
comparing the same to our prior-period reported revenues. Our 
revenues in re ...`


---

### Case 3: What does L stand for in C-LIFE?

**Expected Answer:** `Leadership by example`


#### Fixed-Size Chunking (Failure / Hallucination)

- **Retrieved Chunks Contain Expected Answer:** True

- **LLM Generated Answer:**
```
Answer:
I don't have enough information in the provided documents.

Sources Used:
[Source 5]

Confidence:
Low
```

- **Top Retrieved Chunks:**

  * **Chunk 1:** `... y – –
WongDoody d.o.o (formerly known as oddity code d.o.o)(19)(26)Serbia 100 100
oddity waves GmbH( 18)(26)Germany – –
oddity group services GmbH(18)(26)Germany – –
BASE life science A/S(12)Denmark 100 100
BASE life science AG(21)Switzerland 100 100
BASE life science GmbH(21)Germany 100 100
BASE li ...`

  * **Chunk 2:** `... ly 100 100
Innovisor Inc.(21)US 100 100
BASE life science Inc.(21)US 100 100
BASE life science S.L.( 21)Spain 100 100
InSemi Technology Services Private Limited(30)India 100 100
Elbrus Labs Private Limited(30)(22)India 100 –
Infosys Services (Thailand) Limited(1)(32)Thailand 100 – ...`


#### Recursive Chunking (Success)

- **Retrieved Chunks Contain Expected Answer:** True

- **LLM Generated Answer:**
```
Answer:
L stands for **Leadership** (as in “Leadership by example”) in the C‑LIFE acronym.

Sources Used:
[2]

Confidence:
High
```

- **Top Retrieved Chunks:**

  * **Chunk 1:** `...  GmbH(18)(26)Germany – –
WongDoody d.o.o (formerly known as oddity code d.o.o)(19)(26)Serbia 100 100 oddity waves GmbH(18)(26)Germany – –
oddity group services GmbH(18)(26)Germany – –
BASE life science A/S(12)Denmark 100 100
BASE life science AG(21)Switzerland 100 100
BASE life science GmbH(21)Germa ...`

  * **Chunk 2:** `... development of businesses and 
communities. We reaffirmed our 
long-standing focus across core areas including climate change, nurturing 
workplace inclusivity, employee 
wellness and experience, amplifying 
communities, corporate governance, 
data privacy and information 
management.Our Values 
Our ...`


---
