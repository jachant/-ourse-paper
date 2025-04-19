import json
import pandas as pd
from openai import OpenAI
import evaluate
from typing import Union
from  datasets import Dataset

class CodeProcessor:
    def __init__(self,
                 dataset: Union[pd.DataFrame, Dataset],  # Принимаем оба типа
                 code_column: str = 'code',
                 text_column: str = 'text',
                 model_name: str = "meta-llama/llama-4-maverick",
                 api_key: str = None):
        """
        Инициализация обработчика кода

        :param dataset: Может быть pandas DataFrame или Hugging Face Dataset
        """
        # Конвертируем Dataset в pandas DataFrame если нужно
        if isinstance(dataset, Dataset):
            self.dataset = dataset.to_pandas()
        else:
            self.dataset = dataset.copy()  # Для DataFrame

        self.code_col = code_column
        self.text_col = text_column
        self.model_name = model_name

        # Остальная часть инициализации без изменений
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://your-site.com",
                "X-Title": "Code Processing Tool"
            }
        )

        self.system_message = {
            "role": "system",
            "content": """For each code snippet provided as input..."""  # Полный текст как было
        }

        # Инициализация метрик
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bleurt = evaluate.load('bleurt', 'bleurt-large-512')
        self.bertscore = evaluate.load("bertscore")

    def rename_columns(self, new_code_name: str, new_text_name: str) -> None:
        """Переименование столбцов с кодом и текстом"""
        self.dataset = self.dataset.rename(columns={
            self.code_col: new_code_name,
            self.text_col: new_text_name
        })
        self.code_col = new_code_name
        self.text_col = new_text_name

    def generate_summaries(self, start_idx: int, end_idx: int) -> tuple:
        """Генерация кратких описаний для диапазона записей"""
        codes = ""
        references = []

        for idx in range(start_idx, end_idx):
            if idx >= len(self.dataset):
                break

            sample = self.dataset.iloc[idx]
            codes += sample[self.code_col] + "\n$\n"
            references.append(sample[self.text_col])

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                self.system_message,
                {"role": "user", "content": codes}
            ]
        )

        summaries = completion.choices[0].message.content.split('$')
        return summaries[:len(references)], references

    def save_results(self, summaries: list, references: list,
                     start_idx: int, end_idx: int, filename: str) -> None:
        """Сохранение результатов в JSON файл"""
        output_data = []

        for i in range(len(summaries)):
            if (start_idx + i) >= len(self.dataset):
                break

            entry = {
                "code": self.dataset.iloc[start_idx + i][self.code_col],
                "ideal_answer": references[i],
                "generated_answer": summaries[i].strip()
            }
            output_data.append(entry)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def calculate_metrics(self, predictions: list, references: list) -> dict:
        """Вычисление всех метрик качества"""
        metrics = {}

        # ROUGE-L
        metrics['rouge'] = self.rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rougeL"]
        )["rougeL"]

        # BLEU
        metrics['bleu'] = self.bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )

        # BLEURT
        bleurt_scores = self.bleurt.compute(
            predictions=predictions,
            references=references
        )["scores"]
        metrics['bleurt'] = {
            'scores': bleurt_scores,
            'average': sum(bleurt_scores) / len(bleurt_scores)
        }

        # BERTScore
        bert_scores = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type="microsoft/deberta-large-mnli"
        )
        metrics['bertscore'] = {
            'precision': bert_scores["precision"],
            'recall': bert_scores["recall"],
            'f1': bert_scores["f1"],
            'average_f1': sum(bert_scores["f1"]) / len(bert_scores["f1"])
        }

        return metrics

    def process_batch(self, start_idx: int, end_idx: int, output_file: str) -> dict:
        """Полный цикл обработки батча данных"""
        summaries, references = self.generate_summaries(start_idx, end_idx)
        self.save_results(summaries, references, start_idx, end_idx, output_file)
        return self.calculate_metrics(summaries, references)