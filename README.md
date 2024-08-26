# BERT-NER-Bank-Transactions
This repository features a BERT-based Named Entity Recognition (NER) model fine-tuned for extracting key entities from bank transaction SMS messages. It identifies entities like bank names, amounts, dates, and account numbers. The implementation includes data preprocessing, model training, and evaluation using Hugging Face's Transformers library.


BERT for Named Entity Recognition in Bank Transactions
=======================================================

Overview
--------
This repository contains a BERT-based Named Entity Recognition (NER) model fine-tuned for extracting key entities from bank transaction SMS messages. The model identifies entities such as `BANK`, `MONEY`, `DATE`, `TIME`, `ACCOUNT_NUMBER`, and `TRANSACTION_TYPE`.

Installation
------------
### Prerequisites
- Python 3.6+
- Required libraries: `transformers`, `datasets`, `pandas`, `torch`

You can install the required packages using pip:
pip install transformers datasets pandas torch

### Clone the Repository
To get started, clone the repository using the following commands:

git clone https://github.com/yourusername/BERT-NER-Bank-Transactions.git
cd BERT-NER-Bank-Transactions


Dataset
-------
The dataset used in this project consists of annotated bank transaction SMS messages. Each message is tokenized, and each token is labeled with its corresponding entity.

Model Training
--------------
### Tokenization
The SMS messages are tokenized using `BertTokenizerFast`:

```python
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')


### Tokenization
The model is fine-tuned using the training dataset with the following code:
from transformers import BertForTokenClassification, Trainer, TrainingArguments

model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label_list))

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()


## Usage
inputs = tokenizer("Your SMS message here", return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)
