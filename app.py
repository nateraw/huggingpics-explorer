import logging
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
import streamlit as st
from huggingface_hub import Repository, create_repo, login, whoami
from huggingpics.data import get_image_urls_by_term
from requests.exceptions import HTTPError
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def show_images_of_term(search_term, num_cols=5, num_rows=3):

    # Get the image urls
    # Arbitrarily adding 2 to make sure we have enough images in the event of a failed request
    urls = get_image_urls_by_term(search_term, count=(num_rows * num_cols) + 2)

    st.title(search_term)
    for row_id in range(num_rows):
        cols = st.columns(num_cols)
        for col_id in range(num_cols):
            cols[col_id].image(urls[row_id * num_cols + col_id], use_column_width=True)


def download_image(img_url, filename):
    response = requests.get(img_url)
    response.raise_for_status()
    img_bytes = response.content
    with open(filename, 'wb') as img_file:
        img_file.write(img_bytes)


def make_huggingpics_imagefolder(data_dir, search_terms, count=150, overwrite=False, resume=False, streamlit=False):

    data_dir = Path(data_dir)

    if data_dir.exists():
        if overwrite:
            print(f"Deleting existing HuggingPics data directory to create new one: {data_dir}")
            shutil.rmtree(data_dir)
        else:
            print(f"Using existing HuggingPics data directory: '{data_dir}'")
            if not resume:
                return

    if streamlit:
        pbar = st.progress(0)

    for search_term_idx, search_term in enumerate(search_terms):
        search_term_dir = data_dir / search_term

        search_term_dir.mkdir(exist_ok=True, parents=True)
        is_term_dir_nonempty = any(Path(search_term_dir).iterdir())
        if is_term_dir_nonempty:
            print(f"Skipping search term '{search_term}' because it already has images in it.")
            continue

        urls = get_image_urls_by_term(search_term, count)
        logger.info(f"Saving images of {search_term} to {str(search_term_dir)}...")

        with ThreadPoolExecutor() as executor:
            for i, url in enumerate(tqdm(urls)):
                executor.submit(download_image, url, search_term_dir / f'{i}.jpg')

        if streamlit:
            pbar.progress((search_term_idx + 1) / len(search_terms))

    if streamlit:
        pbar.empty()


def zip_imagefolder(data_dir, zip_path='images.zip'):
    data_dir = Path(data_dir)
    zip_file = zipfile.ZipFile(zip_path, 'w')
    for img_path in data_dir.glob('**/*.jpg'):
        zip_file.write(img_path, arcname=f"{img_path.parent.name}/{img_path.name}")
    zip_file.close()


def get_search_terms():
    terms = [
        st.sidebar.text_input("Term 1:"),
    ]
    while terms[-1] != "":
        terms.append(
            st.sidebar.text_input(
                f"Term {len(terms) + 1}:",
            )
        )
    terms = terms[:-1]
    return terms


def main():

    with st.sidebar:
        st.title('🤗🖼 HuggingPics Explorer')
        st.markdown(
            """
            <p align="center">
                <a href="https://github.com/nateraw/huggingpics-explorer" alt="Repo"><img src="https://img.shields.io/github/stars/nateraw/huggingpics-explorer?style=social" /></a>
            </p>
            """,
            unsafe_allow_html=True,
        )

    names = get_search_terms()
    for name in names:
        show_images_of_term(name)

    with st.sidebar:
        with st.form("Upload to 🤗 Hub"):
            username = st.text_input('Username')
            password = st.text_input('Password', type="password")
            dataset_name = st.text_input('Dataset Name', value='huggingpics-data')
            submit = st.form_submit_button('Upload to 🤗 Hub')
        if submit:
            try:
                token = login(username, password)
                repo_url = create_repo(dataset_name, token, exist_ok=True, repo_type='dataset')
                with TemporaryDirectory() as tmp_dir:
                    repo_owner, repo_name = username, dataset_name
                    repo_namespace = f"{repo_owner}/{repo_name}"

                    repo = Repository(
                        tmp_dir,
                        clone_from=repo_url,
                        use_auth_token=token,
                        git_user=username,
                        git_email=f'{username}@users.noreply.huggingface.co',
                        repo_type='dataset',
                    )
                    temp_path = Path(tmp_dir)
                    imagefolder_path = temp_path / 'images/'
                    zipfile_path = temp_path / 'images.zip'
                    with st.spinner(f"Uploading files to [{repo_namespace}]({repo_url})..."):

                        with repo.commit("Uploaded from HuggingPics Explorer"):
                            make_huggingpics_imagefolder(
                                imagefolder_path, names, count=20, overwrite=True, resume=False, streamlit=True
                            )
                            zip_imagefolder(imagefolder_path, zipfile_path)

                st.success(f"View your dataset here 👉 [{repo_namespace}]({repo_url})")
            except HTTPError as e:
                st.error("Invalid username or password.")


if __name__ == '__main__':
    main()
