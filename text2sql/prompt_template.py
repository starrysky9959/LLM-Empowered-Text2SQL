SQL_SYSTEM_PROMPT = """You are an experienced and professional database administrator. Given the [Schema] about the database, your task is to write a [SQL] to answer the [Question]. In the [Schema], each table consists of several columns and each line describes the name and type of the column. Some external knowledge about the [Schema] and [Question] is provided in the [Evidence]. 
Attention please, [SQL] should satisfy the following constraints:
- In `SELECT <column>`, must only use the column given in the [Schema].
- In `FROM <table>` or `JOIN <table>`, must only use the table given in the [Schema].
- In `JOIN`, must only use the columns with foreign key references in the [Schema].
- Without any specific instruction, use `ASC` for `ORDER BY` by default.
- Consider using `DISTINCT` when you need to eliminate duplicates.
- The content in quotes is case sensitive.
- Prioritize columns whose value are more relevant to the [Question].
""".lstrip()

SQL_INSTRUCTION_PATTERN = """
[Schema]
{schema}
[Question]
{question}
[Evidence]
{evidence}
[Error SQL]
{error_sql}
[Error Message]
{error_message}
[SQL]
""".lstrip()

SQL_OUTPUT_PATTERN = """
{sql}
""".strip()

KEYWORDS_EXTRACT_PATTERN = """
[Objective]
Analyze the given [Question] and [Evidence] to identify and extract keywords. These elements are crucial for understanding the core components of the inquiry and the guidance provided.
[Question]
{question}
[Evidence]
{evidence}
[Output Format]
Please only provide the keywords separated by a comma, no explanations needed.
""".lstrip()
