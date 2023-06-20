<h2>Genocide Transcript Corpus (GTC)</h2>

The Genocide Transcript Corpus (GTC) provides transcript data from three different genocide tribunals: the Extraordinary Chambers in the Courts of Cambodia (ECCC), the International Criminal Tribunal for Rwanda (ICTR), and the International Criminal Tribunal for the Former Yugoslavia (ICTY). 

**GTC Version 2 - June 2023**

Besides meta data regarding the respective tribunal and transcript annotation this version also includes the annotation of text segments that inlude potentially traumatic witness experiences.

The updated version of the GTC contains 52,845 text segments of a total of 90 transcripts that can be attributed to an individual person or court proceedings. The final data set includes the following variables:
* **Case** information: tribunal, case number, accused
* **Transcript** information: document ID, url-link to the original
transcript, date
* **Witness** information: witness name or pseudonym, number
of witnesses per transcript
* **Text** information: speaker (e.g., Witness, LawyerQA), text,
trauma label
* **Annotation** information: annotation ID, start ID, and document
ID


*Codebook V2*

| Variable Name  | Description |
| ------------- | ------------- |
| paragrah | A text passage from a genocide tribunal transcript (250 words each). |
| trauma | Potentially trauma-related content: <br> <ul> <li>not containing trauma-related content = 0</li> <li>containing trauma-related content = 1</li> </ul> |
| tribunal  | Name of the tribunal (ICTY, ICTR, or ECCC)|
| witness | The witness's name or a pseudonym. |
| document | The document number / ID. |
| case | The case number / ID. |
| date | The trial date. |



Please refer to the corresponding paper for further context, including details on the labeling process:

Miriam Schirmer, Isaac Misael Olguín Nolasco, Edoardo Mosca, Shanshan Xu, and Jürgen Pfeffer. 2023. Uncovering Trauma in Genocide Tribunals: An NLP Approach Using the Genocide Transcript Corpus. In Nineteenth International Conference on Artificial Intelligence and Law (ICAIL 2023), June 19–23, 2023, Braga, Portugal. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3594536.3595147



**GTC Version 1 - June 2022**

All  samples  were  labeled  according  to  whether  they contain a witness’s description of experienced violence. Violence in this context includes accounts of experienced torture, interrogation, death, beating, psychological violence, experienced military attacks, destruction of villages, and looting. 

The transcript data was divided into equally large text chunks of 250 words each. Numbers  and  punctuation  were  removed.

*Codebook V1*

| Variable Name  | Description |
| ------------- | ------------- |
| paragraph | A text passage from a genocide tribunal transcript (250 words each). |
| label | Violence-related content: <br> <ul> <li>not containing violence = 0</li> <li>containing violence = 1</li> </ul> |
| tribunal  | The specific tribunal the transcript data is from: <br> <ul> <li>ECCC = 1</li>  <li>ICTY = 2</li> <li>ICTR = 3 </li> </ul> |
| witness | The witness's name or a pseudonym. |
| document | The document number / ID. |
| case | The case number / ID. |
| date | The trial date. |


All of the used transcripts are openly accessible on the respective courts' websites.
