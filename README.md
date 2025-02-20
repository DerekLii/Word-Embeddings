### ---Setup---

1. Download the code, extract it, and run it in VSCode
2. Open up the terminal, use Bash
3. pip install datasets
4. pip install apache_beam
5. pip install gensim
6. pip install nltk
7. pip install wefe
8. go to https://vectors.nlpl.eu/repository/ and download ID 11 Gigaword 5th Edition and ID 12 Gigaword 5th Edition, rename them model11.bin and model12.bin and place them in the folder. These files are too large so I can't upload them to Github, they take about 3-5 minutes to download.
9. Direct download links if you need them: https://vectors.nlpl.eu/repository/20/11.zip and https://vectors.nlpl.eu/repository/20/12.zip
10. Make sure you're in the right folder before running the program command below.

### ---Run the program---

python word_embeddings.py

The next 3 require model11.bin and model12.bin to be in your folder, as those are the pre-trained models I imported.

python compare.py

python weat.py

python text_classification.py

### ---Results---

![image](https://github.com/user-attachments/assets/235f7d82-c684-45e0-8083-a9a46347b4bf)

![image](https://github.com/user-attachments/assets/72e7b2c2-350b-4396-8227-43be5d1789d5)

![image](https://github.com/user-attachments/assets/42cec81a-8d50-4f98-99c3-cbe1cf6a898b)

![image](https://github.com/user-attachments/assets/9027de4c-d333-41c4-997d-c498d3f1df50)

