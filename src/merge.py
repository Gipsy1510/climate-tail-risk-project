import pyarrow.parquet as pq

table = pq.ParquetDataset([
    "data/news_article_text_p1_2015_2017.parquet",
    "data/news_article_text_p2_2018_2020.parquet",
    "data/news_article_text_p3_2021_2023.parquet",
    "data/news_article_text_p4_2024_2025.parquet"
]).read()

pq.write_table(table, "data/merged.parquet")