import logging
import shutil
import time
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


def make_huggingpics_imagefolder(data_dir, search_terms, count=150, overwrite=False, transform=None, resume=False):

    data_dir = Path(data_dir)

    if data_dir.exists():
        if overwrite:
            print(f"Deleting existing HuggingPics data directory to create new one: {data_dir}")
            shutil.rmtree(data_dir)
        else:
            print(f"Using existing HuggingPics data directory: '{data_dir}'")
            if not resume:
                return

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

        pbar.progress((search_term_idx + 1) / len(search_terms))

    pbar.empty()


def create_dataset(terms):

    msg_placeholder = st.empty()

    for term in terms:
        show_images_of_term(term)

    with st.sidebar:
        with st.form('Push to Hub'):
            dataset_name = st.text_input('Dataset Name', value='huggingpics-data')
            do_push = st.form_submit_button("Push to 🤗 Hub")

    if do_push:
        msg_placeholder.empty()
        if not st.session_state.get('is_logged_in'):
            msg_placeholder.error("You must login to push to the hub.")
            return
        else:
            msg_placeholder.empty()

        with st.sidebar:
            repo_url = create_repo(dataset_name, st.session_state.token, exist_ok=True, repo_type='dataset')
            hf_username = whoami(st.session_state.token)['name']
            with TemporaryDirectory() as tmp_dir:
                repo_owner, repo_name = hf_username, dataset_name
                repo_namespace = f"{repo_owner}/{repo_name}"
                repo_url = f'https://huggingface.co/datasets/{repo_namespace}'

                repo = Repository(
                    tmp_dir,
                    clone_from=repo_url,
                    use_auth_token=st.session_state.token,
                    git_user=hf_username,
                    git_email=f'{hf_username}@users.noreply.huggingface.co',
                    repo_type='dataset',
                )

                with st.spinner(f"Uploading files to [{repo_namespace}]({repo_url})..."):
                    with repo.commit("Uploaded from HuggingPics Explorer"):
                        make_huggingpics_imagefolder(Path(tmp_dir) / 'images', terms, count=150)

                st.success(f"View your dataset here 👉 [{repo_namespace}]({repo_url})")


def huggingface_auth_form():
    placeholder = st.empty()

    is_logged_in = st.session_state.get('is_logged_in', False)

    if is_logged_in:
        token = st.session_state.token
        with placeholder.container():
            st.markdown(f"✅ Logged in as {whoami(token)['name']}")
            do_logout = st.button("Logout")
        if do_logout:
            st.session_state.token = None
            st.session_state.is_logged_in = False
            placeholder.empty()
            huggingface_auth_form()
    else:
        with placeholder.container():
            username = st.text_input('Username', value=st.session_state.get('username', ''))
            password = st.text_input('Password', value="", type="password")
            submit = st.button('Login')
        if submit:
            try:
                st.session_state.token = login(username, password)
                st.session_state.is_logged_in = True
                placeholder.empty()
                huggingface_auth_form()
            except HTTPError as e:
                st.session_state.token = None
                st.session_state.is_logged_in = False
                st.error("Invalid username or password.")
                time.sleep(2)
                # huggingface_auth_form()  # ???


def main():

    with st.sidebar:
        st.markdown(
            """
            <p align="center">
                <h1>🤗🖼 HuggingPics Explorer</h1>
            <p/>
            <p align="center">
                <a href="https://github.com/nateraw/huggingpics-explorer" alt="Repo"><img src="https://img.shields.io/github/stars/nateraw/huggingpics-explorer?style=social" /></a>
            </p>
            """,
            unsafe_allow_html=True,
        )

        term_1 = st.sidebar.text_input('Search Term 1', value='shiba inu')
        term_2 = st.sidebar.text_input('Search Term 2', value='husky')
        term_3 = st.sidebar.text_input('Search Term 3', value='')
        term_4 = st.sidebar.text_input('Search Term 4', value='')
        term_5 = st.sidebar.text_input('Search Term 5', value='')
        terms = [t for t in [term_1, term_2, term_3, term_4, term_5] if t]

        st.markdown('---')
        huggingface_auth_form()
        st.markdown('---')

    _ = create_dataset(terms)


if __name__ == '__main__':
    main()
