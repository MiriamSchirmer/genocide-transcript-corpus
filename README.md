<h2>Genocide Transcript Corpus (GTC)</h2>

The Genocide Transcript Corpus (GTC) provides transcript data from three different genocide tribunals: the Extraordinary Chambers in the Courts of Cambodia (ECCC), the International Criminal Tribunal for Rwanda (ICTR), and the International Criminal Tribunal for the Former Yugoslavia (ICTY). 

All  samples  were  labeled  according  to  whether  they contain a witness’s description of experienced violence. Violence in this context includes accounts of experienced torture, interrogation, death, beating, psychological violence, experienced military attacks, destruction of villages, and looting. 

The transcript data was divided into equally large text chunks of 250 words each. Numbers  and  punctuation  were  removed.

*Codebook*

| Variable Name  | Description |
| ------------- | ------------- |
| paragrah | A text passage from a genocide tribunal transcript (250 words each). |
| label | Violence-related content: <br> <ul> <li>not containing violence = 0</li> <li>containing violence = 1</li> </ul> |
| tribunal  | The specific tribunal the transcript data is from: <br> <ul> <li>ECCC = 1</li>  <li>ICTY = 2</li> <li>ICTR = 3 </li> </ul> |
| witness | The witness's name or a pseudonym. |
| document | The document number / ID. |
| case | The case number / ID. |
| date | The trial date. |

All of the used transcripts are openly accessible on the respective courts' websites.
