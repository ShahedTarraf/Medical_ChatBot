import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

# تحميل متغيرات البيئة
load_dotenv()
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# تعديل RTL للغة العربية
st.markdown(
    """
    <style>
    html, body, .main {
        direction: rtl;
        text-align: right;
    }
    .st-chat-message > div {
        direction: rtl;
        text-align: right;
    }
    .stTextInput>div>input {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# تحميل قاعدة FAISS
@st.cache_resource
def get_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding, allow_dangerous_deserialization=True)
    return db

# قائمة التحيات
GREETINGS = ["مرحبا", "أهلاً", "أهلا", "هاي", "السلام عليكم"]

# Prompt للبوت الذكي + سؤال متابعة
retrieval_prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
أنت مساعد طبي ذكي جدًا، دقيق، منطقي، وقادر على الفهم والتفسير. مهمتك:

1. إذا كان السؤال الطبي موجودًا في السياق (context)، أجب بطريقة واضحة ومقنعة، حتى لو كان السؤال مختصرًا أو استخدم كلمات مشابهة.
2. إذا لم يكن السؤال موجودًا نصيًا في المستندات، حاول شرح الإجابة بطريقة علمية وآمنة بناءً على المعرفة الطبية العامة، مع الإشارة إلى أنها من المعرفة العامة وليست من المستندات.
3. اجعل إجابتك مفهومة للمستخدم العادي، قصيرة وواضحة ومقنعة.
4. لا تختلق معلومات خطيرة أو مضللة.
5. بعد كل إجابة، اقترح سؤال متابعة طبي قصير وملائم للمستخدم.
6. إذا لم تعرف الإجابة، أجب: "لا توجد معلومات متاحة في المستندات أو المعرفة الطبية العامة".

السياق (المستندات):
{context}

سؤال المستخدم:
{input}

إجابة (مع اقتراح سؤال متابعة في النهاية):
"""
)

def main():
    st.title("🤖 مساعدك الطبي الذكي")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # عرض المحادثات السابقة
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_prompt = st.chat_input("اكتب سؤالك أو تحيتك هنا...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # الرد على التحيات بشكل مناسب
        if "صباح الخير" in user_prompt:
            greeting_reply = "صباح النور! كيف يمكنني مساعدتك اليوم؟"
        elif "مساء الخير" in user_prompt:
            greeting_reply = "مساء النور! كيف يمكنني مساعدتك اليوم؟"
        elif any(greet in user_prompt for greet in GREETINGS):
            greeting_reply = "مرحبا! كيف يمكنني مساعدتك اليوم؟"
        else:
            greeting_reply = None

        if greeting_reply:
            st.chat_message("assistant").markdown(greeting_reply)
            st.session_state.messages.append({"role": "assistant", "content": greeting_reply})
        else:
            try:
                db = get_vectorstore()
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.0,
                    max_tokens=512,
                    api_key=GROQ_API_KEY
                )

                # إنشاء سلسلة RAG مع البحث الموسع
                combine_chain = create_stuff_documents_chain(llm, retrieval_prompt)
                rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k": 10}), combine_chain)

                # استدعاء RAG
                result = rag_chain.invoke({"input": user_prompt})

                # استخراج النص الفعلي فقط
                if isinstance(result, dict):
                    answer = result.get("output_text") or result.get("text")
                elif hasattr(result, "content"):
                    answer = result.content
                else:
                    answer = str(result)

                # إذا لم توجد نتيجة من PDF → نستخدم LLM للمعرفة العامة
                if not answer or "لا توجد معلومات متاحة" in answer:
                    llm_response = llm.invoke(
                        f"أجب على السؤال الطبي التالي بطريقة علمية وواضحة للمستخدم العادي. "
                        f"إذا لم تعرف الإجابة بدقة، أجب 'لا توجد معلومات متاحة': {user_prompt}"
                    )
                    if hasattr(llm_response, "content"):
                        answer = llm_response.content
                    else:
                        answer = str(llm_response)
                    answer = f"ملاحظة: هذه المعلومات من المعرفة العامة. {answer}"

                # عرض الإجابة
                st.chat_message("assistant").markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"❌ خطأ: {str(e)}")

if __name__ == "__main__":
    main()
