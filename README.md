cd .\venv\Scripts\
activate
cd ..
cd ..

npx inngest-cli@latest dev
uvicorn main:app --reload
streamlit run streamlit_app.py