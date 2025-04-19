import myClass
import prepare
processor = myClass.CodeProcessor(
    dataset=prepare.dataset_xlcost,
    code_column="code",
    text_column="text",
    model_name="deepseek/deepseek-chat-v3-0324:free",
    api_key="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"
)
print(1)
processor.generate_summaries(0, 20)
print(2)
processor.save_results(
    start_idx=0,
    end_idx=20,
    filename="deepseek-xlcost"
)