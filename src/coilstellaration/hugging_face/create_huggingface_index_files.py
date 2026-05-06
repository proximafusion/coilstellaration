import os
import pathlib
import tempfile

import duckdb  # pyright: ignore[reportMissingImports]
from huggingface_hub import upload_file

LOCAL_OUTDIR = pathlib.Path(
    "/home/devuser/tmp/outputs/coilstellaration/huggingface_index_files"
)
LOCAL_OUTDIR.mkdir(parents=True, exist_ok=True)


def create_hf_index(folder: str, repo_id: str, upload: bool = False):
    print(f"Creating Hugging Face index for {folder}...")
    with tempfile.TemporaryDirectory(suffix=f"_{folder}") as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)
        tmpdir.mkdir(parents=True, exist_ok=True)

        db_path = tmpdir / "index.db"

        print(f"Connecting to temporary DuckDB at {db_path}...")
        con = duckdb.connect(db_path)
        hf_token = os.environ["HF_TOKEN"]
        con.execute(f"""
            CREATE SECRET hf_secret (
                TYPE HUGGINGFACE,
                TOKEN '{hf_token}'
            );
            PRAGMA enable_progress_bar;
            PRAGMA progress_bar_time = 1000;
        """)

        print("Building the mapping table in DuckDB by reading remote Parquet files...")
        build_query = f"""
            CREATE TABLE id_mapping AS
            SELECT
                id,
                path
            FROM read_parquet(
                'hf://datasets/{repo_id}/{folder}/*.parquet',
                filename='path'
                )
            WHERE path NOT LIKE '%id_to_file_map.parquet%'
        """
        con.execute(build_query)

        print("Creating an index on the id column for microsecond lookups later...")
        con.execute("CREATE UNIQUE INDEX id_index ON id_mapping (id)")

        id_to_file_map_path = LOCAL_OUTDIR / folder / "id_to_file_map.parquet"
        id_to_file_map_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Exporting the id to file mapping from DuckDB to {id_to_file_map_path}..."
        )
        export_query = f"""
            COPY (
                SELECT
                    id,
                    REPLACE(path, 'hf://datasets/{repo_id}/', '') AS path
                FROM id_mapping
            ) TO '{id_to_file_map_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
        con.execute(export_query)

        print(f"Successfully exported the database to {id_to_file_map_path}!")
        con.close()

        if upload:
            print(f"Uploading {id_to_file_map_path} to Hugging Face...")
            upload_file(
                path_or_fileobj=id_to_file_map_path,
                repo_id=repo_id,
                path_in_repo=f"{folder}/id_to_file_map.parquet",
                repo_type="dataset",
            )


if __name__ == "__main__":
    print("Building local index... This will take a few minutes.")
    create_hf_index("vmecpp_wout", "proxima-fusion/constellaration")
    create_hf_index("metrics", "proxima-fusion/coilstellaration", upload=True)
    create_hf_index("requirements", "proxima-fusion/coilstellaration", upload=True)
    create_hf_index("coilsets", "proxima-fusion/coilstellaration", upload=True)
