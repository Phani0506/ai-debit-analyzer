# Entire Updated Script
import streamlit as st
import PyPDF2
from groq import Groq
import io
import re
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from collections import Counter, defaultdict
import random
import math
import time
import httpx

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Debit Analyzer AI")

# --- Global Configuration & Constants ---
try:
    raw_debug_mode = st.secrets.get("DEBUG_MODE", False)
    IS_DEBUG_MODE = str(raw_debug_mode).lower() == 'true' if isinstance(raw_debug_mode, str) else bool(raw_debug_mode)
except Exception:
    IS_DEBUG_MODE = False

if IS_DEBUG_MODE:
    st.sidebar.warning("⚠️ DEBUG MODE IS ON (UI logs visible) ⚠️")
else:
    st.sidebar.info("App is in standard mode (UI debug logs hidden).")

# --- API KEY CONFIGURATION & Client Initialization ---
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = None
    st.session_state.api_key_initialized = False
    st.session_state.api_key_valid = False

if not st.session_state.api_key_initialized:
    try:
        api_key_from_secrets = st.secrets.get("GROQ_API_KEY")
        if not api_key_from_secrets:
            st.error("Groq API key (GROQ_API_KEY) not found/empty in secrets. App owner: Configure it.")
        else:
            client = Groq(api_key=api_key_from_secrets, timeout=90.0)
            client.models.list()
            st.session_state.groq_client = client
            st.session_state.api_key_valid = True
            if IS_DEBUG_MODE: st.success("Groq API key loaded and client initialized successfully!")
    except Exception as e:
        st.error(f"Failed to initialize Groq client or validate key: {e}")
    finally:
        st.session_state.api_key_initialized = True

GROQ_MODEL_EXTRACTION = "llama3-8b-8192"
GROQ_MODEL_CATEGORIZATION = "llama3-8b-8192"
CLIENT_INSTANCE = st.session_state.groq_client

CURRENCY_SYMBOL = "₹"
INCOME_CATEGORIES = ["Income", "Salary", "Freelance Income", "Investment Income"]
COMMON_CATEGORIES_FOR_LABELLING = [
    "Groceries", "Utilities", "Rent/Mortgage", "Dining Out", "Transportation",
    "Shopping", "Entertainment", "Healthcare", "Subscriptions", "Travel",
    "Education", "Gifts/Donations", "Fees", "Cash Withdrawal",
    "Other Expense", "Uncategorized"
]
# DEFAULT_CATEGORIES = COMMON_CATEGORIES_FOR_LABELLING[:] # Not strictly needed if COMMON_CATEGORIES_FOR_LABELLING is the master

# --- PDF Processing Functions (unlock_pdf, extract_transactions_from_page_with_llm) ---
# These functions remain the same as the previous version. For brevity, I'll omit them here,
# but assume they are present and working as before.
# --- PDF Processing Functions ---
def unlock_pdf(file_bytes, password):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        if pdf_reader.is_encrypted:
            if password:
                if pdf_reader.decrypt(password):
                    if IS_DEBUG_MODE: st.info("PDF unlocked (debug).")
                    return pdf_reader
                else: st.error("Incorrect password."); return None
            else: st.warning("PDF needs password."); return None
        else:
            if IS_DEBUG_MODE: st.info("PDF not password protected (debug).")
            return pdf_reader
    except Exception as e: st.error(f"Error reading PDF: {e}"); return None

def extract_transactions_from_page_with_llm(page_text, page_number):
    if not CLIENT_INSTANCE:
        if IS_DEBUG_MODE: print("DEBUG: Groq client N/A in extract_transactions_from_page_with_llm")
        return []
    if not page_text or not page_text.strip(): return []
    prompt = f"""Analyze bank statement text from THIS PAGE ONLY and extract transactions.
    For each, identify Date (YYYY-MM-DD or null), Description, Amount (string), Type ("Debit" or "Credit").
    Format as JSON list of objects: {{"date": "...", "description": "...", "amount": "...", "type": "..."}}.
    If no transactions, return empty JSON list: []. Respond ONLY with the valid JSON list.

    Text from Page {page_number}:\n---\n{page_text[:4000]}\n---\nJSON Output:"""
    try:
        response = CLIENT_INSTANCE.chat.completions.create(
            model=GROQ_MODEL_EXTRACTION,
            messages=[
                {"role": "system", "content": "Extract bank transactions into JSON list format ONLY."},
                {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=3000)
        raw_content = response.choices[0].message.content
        if IS_DEBUG_MODE: print(f"DEBUG (Page {page_number} Extraction Raw LLM): {raw_content[:200]}...")
        match = re.search(r'(\[[\s\S]*\])', raw_content)
        if match:
            try:
                data = json.loads(match.group(0))
                if not isinstance(data, list): return []
                valid = []
                for item in data:
                    if isinstance(item, dict) and all(k in item for k in ["description", "amount", "type"]):
                        item_type = item.get("type","").strip().lower()
                        if item_type in ["debit", "credit"]:
                            item["type"] = item_type.capitalize()
                            item["date"] = item.get("date")
                            valid.append(item)
                return valid
            except json.JSONDecodeError:
                if IS_DEBUG_MODE: print(f"P{page_number} JSON ERR. RAW MATCHED:\n{match.group(0)}")
                return []
        else:
            if IS_DEBUG_MODE: print(f"P{page_number} NO JSON LIST IN RAW. RAW:\n{raw_content}")
            return []
    except Exception as e:
        if IS_DEBUG_MODE: st.error(f"Groq P{page_number} extract err (debug): {e}")
        print(f"ERROR: Groq Page {page_number} Exception: {e}")
        return []


# --- Data Processing and Sampling ---
def clean_amount(amount_str):
    if not isinstance(amount_str, str): return None
    cleaned = re.sub(r'[^\d\.]', '', amount_str)
    try: return abs(float(cleaned))
    except: return None

def sample_debit_transactions(transactions):
    debit_tx = [t.copy() for t in (transactions or []) if t.get("type") == "Debit"]
    for t in debit_tx:
        t['parsed_amount'] = clean_amount(t.get("amount"))
        t['id'] = id(t) # Assign a unique ID based on object's memory address for this session
    debit_tx = [t for t in debit_tx if t['parsed_amount'] is not None]

    if not debit_tx:
        if IS_DEBUG_MODE: st.warning("Debug: No valid debit transactions for sampling.")
        return [], []
    target_samples = min(15, len(debit_tx)) # Increased to 15
    if not target_samples: return [], debit_tx

    labeling_samples = random.sample(debit_tx, target_samples) if len(debit_tx) > target_samples else debit_tx[:]
    if IS_DEBUG_MODE: st.info(f"Debug: Selected {len(labeling_samples)} debit txns for labeling.")
    
    # Ensure IDs are present for sampled transactions
    for t in labeling_samples:
        if 'id' not in t: t['id'] = id(t)

    labeling_ids = {t['id'] for t in labeling_samples}
    remaining_debits = [t for t in debit_tx if t['id'] not in labeling_ids]
    return labeling_samples, remaining_debits


# --- LLM for Categorization ---
def get_debit_categorization_with_llm(user_labeled_examples_list, transactions_to_categorize, available_categories_for_ai):
    # ... (This function remains the same as the previous robust version with DEBUG expanders) ...
    # Ensure it returns a dictionary mapping transaction id(object) to category string
    if not CLIENT_INSTANCE: st.error("Groq client unavailable for categorization."); return {}
    if not transactions_to_categorize:
        if IS_DEBUG_MODE: st.info("Debug: No remaining debits to auto-categorize.")
        return {}
    if not available_categories_for_ai: st.error("Internal Error: No categories for AI."); return {t['id']: "Error - No Categories Defined" for t in transactions_to_categorize}

    system_message = f"""You are an AI assistant that categorizes bank debit transactions.
Your primary goal is to accurately assign a category to a list of new transactions based on user-provided examples.
Carefully analyze the patterns in descriptions and approximate amounts from the user's examples.
You MUST use these examples as your main guide for categorizing the new transactions.
You MUST ONLY choose categories from the following allowed list: {', '.join(available_categories_for_ai)}.
Your response MUST be a numbered list of category names, one for each new transaction, in the exact order they are presented.
Example Response Format (if 3 transactions were sent):
1. Groceries
2. Transportation
3. Shopping
Do not add any other text, explanations, or apologies. Your output should be ONLY the numbered list.
"""
    user_prompt_parts = []
    if user_labeled_examples_list:
        examples_str = "=== USER-LABELED EXAMPLES (Description, Amount -> Category) ===\n"
        examples_str += "These are crucial examples of how the user categorizes their spending. Learn from them:\n"
        for t_dict, cat_str in user_labeled_examples_list: # t_dict is the transaction object
            desc = t_dict.get('description', 'N/A'); amt = t_dict.get('parsed_amount', 0.0)
            examples_str += f"- User Example: Description '{desc}', Amount approx {CURRENCY_SYMBOL}{amt:.2f} was labeled as '{cat_str}'.\n"
        user_prompt_parts.append(examples_str)
    else:
        user_prompt_parts.append("No user examples provided. Use general financial knowledge and common sense, choosing from the allowed categories for the transactions below.\n")

    transactions_list_str = "=== NEW TRANSACTIONS TO CATEGORIZE ===\nBased *very carefully* on the patterns observed in the user examples (if provided), and considering both description and amount similarities, categorize each of the following new debit transactions. If no examples are provided, use your best judgment.\n\n"
    valid_transactions_for_prompt = []
    for i, t_dict in enumerate(transactions_to_categorize):
        desc = t_dict.get('description'); amt = t_dict.get('parsed_amount', 0.0)
        if desc:
            transactions_list_str += f"{i+1}. Description: '{desc}', Amount: {CURRENCY_SYMBOL}{amt:.2f}\n"
            valid_transactions_for_prompt.append(t_dict)

    if not valid_transactions_for_prompt:
        if IS_DEBUG_MODE: st.info("Debug: No valid transactions for categorization prompt.")
        return {}
    user_prompt_parts.append(transactions_list_str)
    user_prompt_parts.append(f"\n=== YOUR RESPONSE (Numbered List of Categories) ===\nRespond with exactly {len(valid_transactions_for_prompt)} category names from the allowed list, corresponding to the transactions above. Ensure each line starts with a number, a dot, and then the category.")
    final_user_prompt = "\n".join(user_prompt_parts)

    if IS_DEBUG_MODE:
        with st.expander("Debug UI: AI Categorization - Remaining Transactions", expanded=False):
            st.text_area("System Message", system_message, height=250, key="d_cat_sys_exp_debug_ui_unique1")
            st.text_area("User Prompt", final_user_prompt, height=400, key="d_cat_usr_exp_debug_ui_unique1")
            st.write("User Labeled Examples (raw list of tuples):", user_labeled_examples_list)
            st.write("Transactions sent to LLM (list of dicts):", valid_transactions_for_prompt)
            st.write("Allowed Categories for LLM:", available_categories_for_ai)
    try:
        response = CLIENT_INSTANCE.chat.completions.create(
            model=GROQ_MODEL_CATEGORIZATION,
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": final_user_prompt}],
            temperature=0.05, max_tokens=len(valid_transactions_for_prompt) * 25 + 200)
        raw_response_content = response.choices[0].message.content
        if IS_DEBUG_MODE:
            with st.expander("Debug UI: AI Categorization - LLM Response", expanded=False):
                st.text_area("LLM Raw Output for Categorization", raw_response_content, height=200, key="d_cat_raw_exp_debug_ui_unique1")
            print(f"DEBUG CONSOLE: Raw LLM Categorization Output:\n{raw_response_content}")

        llm_assigned_categories = []; lines = raw_response_content.strip().split('\n'); parsed_count = 0
        for line in lines:
            s_line = line.strip();
            if not s_line: continue
            match = re.match(r'^\s*\d+\s*[.:]?\s*(.+)', s_line); cat_name = match.group(1).strip() if match else s_line
            if cat_name:
                if cat_name in available_categories_for_ai: llm_assigned_categories.append(cat_name); parsed_count +=1
                else:
                    match_ci = next((c for c in available_categories_for_ai if c.lower() == cat_name.lower()), f"Uncategorized (AI: '{cat_name}')")
                    llm_assigned_categories.append(match_ci)
        if IS_DEBUG_MODE:
            with st.expander("Debug UI: AI Categorization - Parsed Categories", expanded=False):
                st.write(f"Parsed Categories ({len(llm_assigned_categories)} items, {parsed_count} actual categories parsed):", llm_assigned_categories)
        results = {}
        if len(llm_assigned_categories) != len(valid_transactions_for_prompt):
            if IS_DEBUG_MODE: st.warning(f"Debug UI: LLM category count mismatch: Expected {len(valid_transactions_for_prompt)}, Got {len(llm_assigned_categories)}.")
        for i, trans_obj in enumerate(valid_transactions_for_prompt): # trans_obj here is the original dict from transactions_to_categorize
            results[trans_obj['id']] = llm_assigned_categories[i] if i < len(llm_assigned_categories) else "Uncategorized (LLM Response Short)"
        if IS_DEBUG_MODE:
            with st.expander("Debug UI: AI Categorization - Final ID-to-Category Mapping", expanded=False):
                debug_map_display = {f"{t.get('description', 'N/A')[:30]} (ID:{t['id']})": results.get(t['id']) for t in valid_transactions_for_prompt}
                st.write(debug_map_display)
        return results # Returns dict mapping original transaction 'id' to category string
    except Exception as e:
        st.error(f"Groq API categorization error: {e}")
        if IS_DEBUG_MODE: print(f"DEBUG CONSOLE: Categorization Exception: {e}")
        return {t['id']: "Error Categorizing" for t in valid_transactions_for_prompt}


# --- LLM for Suggestions & Plotting ---
# These functions (get_llm_suggestions, create_pie_chart) remain the same.
# Ensure CURRENCY_SYMBOL is used in get_llm_suggestions and create_pie_chart.
def get_llm_suggestions(categorized_expenses_summary):
    if not CLIENT_INSTANCE: st.error("Groq client unavailable for suggestions."); return "Suggestions N/A."
    if not categorized_expenses_summary: return "No expense data for suggestions."
    summary_str = "\n".join([f"- {c}: {CURRENCY_SYMBOL}{a:.2f}" for c, a in categorized_expenses_summary.items() if a > 0])
    if not summary_str: return "No significant expenses for suggestions."
    prompt = f"Monthly debit expenses ({CURRENCY_SYMBOL}):\n{summary_str}\nProvide 2-3 brief, actionable financial suggestions."
    try:
        r = CLIENT_INSTANCE.chat.completions.create(model=GROQ_MODEL_CATEGORIZATION, messages=[{"role": "system", "content": "Helpful financial advisor."}, {"role": "user", "content": prompt}])
        return r.choices[0].message.content
    except Exception as e: st.error(f"Groq suggestions error: {e}"); return "Error generating suggestions."

def create_pie_chart(category_amounts):
    valid_data = {c:a for c,a in category_amounts.items() if a > 0 and not c.startswith("Uncategorized (AI:") and not c.startswith("Uncategorized (LLM") and c not in ["Error Categorizing","Uncategorized","Uncategorized (Mapping Error)", "Error - No Categories Defined"]}
    if not valid_data:
        if IS_DEBUG_MODE: st.info("Debug: No valid spending data to plot in pie chart.");
        return None
    labels = list(valid_data.keys()); sizes = list(valid_data.values())
    fig, ax = plt.subplots(figsize=(10,7)); nc = len(labels)
    if nc == 0: return None
    cmap_name = 'Pastel1' if nc <= 9 else ('Set3' if nc <=12 else 'tab20c'); cmap = plt.cm.get_cmap(cmap_name)
    colors = [cmap(i/nc) for i in range(nc)] if nc > 0 else []
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize':9}, wedgeprops={'edgecolor':'gray'})
    ax.axis('equal'); plt.title(f"Debit Transaction Distribution ({CURRENCY_SYMBOL})", fontsize=12); return fig


# --- Streamlit App UI ---
st.title(f"📄💸 AI Debit Transaction Analyzer ({CURRENCY_SYMBOL})")

# --- Session State Initialization ---
default_state = {
    'app_stage': "upload",
    'extracted_transactions': None,      # Raw from LLM extraction
    'debit_transactions_for_labeling': [], # Sampled debits for user to label
    'remaining_debit_transactions': [],  # Debits not in the sample set
    'user_labeled_examples': [],         # List of (transaction_dict, category_str) from user
    'all_debit_transactions_for_review': [], # Combined list for final review (NEW)
    # 'categorized_results' for AI's output on remaining_debit_transactions is handled directly
    'final_summary': {},
    'llm_insights': "",
    'suggestions_generated_flag': False,
    'ai_suggestions_text_output': ""
}
for k, v in default_state.items():
    if k not in st.session_state: st.session_state[k] = v


# --- STAGE: UPLOAD ---
if st.session_state.app_stage == "upload":
    # ... (Upload UI - same as before, ensure CLIENT_INSTANCE check) ...
    st.header("Step 1: Upload & Extract Transactions")
    if not CLIENT_INSTANCE and not st.session_state.api_key_valid:
        st.error("Groq API Client not initialized. App owner: Check API key in secrets and ensure it's valid.")
        st.stop()
    elif not CLIENT_INSTANCE and st.session_state.api_key_valid:
        st.error("Groq client instance lost. Refresh if issue persists or check API key / service status.")
        st.stop()

    uploaded_file = st.file_uploader("Select PDF bank statement", type="pdf", key="pdf_ul_main_app_v2")
    pdf_password = st.text_input("PDF password (if any)", type="password", key="pdf_pw_main_app_v2")

    if uploaded_file and st.button("Process PDF & Extract", key="proc_btn_main_app_v2", type="primary"):
        # Reset relevant states for a new processing run
        for key_to_reset in default_state.keys():
            if key_to_reset == 'app_stage': continue
            st.session_state[key_to_reset] = default_state[key_to_reset] # Use actual default values
        st.session_state.app_stage = "upload" # Stay until successful

        with st.spinner("Reading PDF..."):
            pdf_reader = unlock_pdf(uploaded_file.getvalue(), pdf_password)

        if pdf_reader:
            # ... (Page-by-page extraction logic - same as before) ...
            num_pages = len(pdf_reader.pages); st.write(f"Processing {num_pages} pages...")
            all_extracted_items = []; progress_bar = st.progress(0); start_time = time.time()
            for i, page_obj in enumerate(pdf_reader.pages):
                current_page_num = i + 1
                progress_bar.progress(i / num_pages, text=f"Page {current_page_num}/{num_pages}: Extracting text & calling AI...")
                try:
                    page_text_content = page_obj.extract_text()
                    if page_text_content and page_text_content.strip():
                        page_transactions_list = extract_transactions_from_page_with_llm(page_text_content, current_page_num)
                        if page_transactions_list:
                            all_extracted_items.extend(page_transactions_list)
                    elif IS_DEBUG_MODE: st.caption(f"Debug: Page {current_page_num}: No text extracted or page is blank.")
                except Exception as page_processing_error:
                    st.error(f"Error processing page {current_page_num}: {page_processing_error}")
            progress_bar.progress(1.0, text="AI Extraction complete for all pages!")
            if IS_DEBUG_MODE: st.write(f"Debug: Total extraction time: {time.time() - start_time:.2f}s. Raw items extracted: {len(all_extracted_items)}")

            if all_extracted_items:
                st.session_state.extracted_transactions = all_extracted_items
                with st.spinner("Filtering debits & selecting samples for labeling..."):
                    labeling_samples, remaining_debit_tx = sample_debit_transactions(all_extracted_items)
                if labeling_samples:
                    st.session_state.debit_transactions_for_labeling = labeling_samples
                    st.session_state.remaining_debit_transactions = remaining_debit_tx
                    st.session_state.app_stage = "labeling"; st.rerun()
                elif remaining_debit_tx: # No samples, but other debits exist
                    st.info("No specific samples were selected for labeling. Proceeding directly to AI categorization for all found debits.")
                    st.session_state.all_debit_transactions_for_review = remaining_debit_tx # All debits go for AI cat then review
                    st.session_state.user_labeled_examples = [] # No user labels in this path
                    st.session_state.debit_transactions_for_labeling = []
                    st.session_state.remaining_debit_transactions = [] # Will be processed by AI then put into review list
                    st.session_state.app_stage = "ai_categorize_all"; st.rerun() # New intermediate stage
                else: st.error("Extraction completed, but no valid debit transactions were found.")
            else: st.error("AI finished processing all pages, but no transactions were extracted.")


# --- STAGE: Labeling User Samples ---
elif st.session_state.app_stage == "labeling":
    # ... (Labeling UI - same as before, ensure keys are unique if duplicated) ...
    st.header("Step 2: Label Sample Debit Transactions")
    if not CLIENT_INSTANCE: st.error("Groq client N/A."); st.stop()
    if not st.session_state.get('debit_transactions_for_labeling'):
        st.warning("No debit transactions were selected for labeling.")
        # ... (buttons for skip/different PDF) ...
        if st.session_state.get('remaining_debit_transactions'):
            if st.button("Skip Labeling & Proceed with AI", type="primary", key="skip_label_btn_main_v2"):
                st.session_state.app_stage = "ai_categorize_all" # Go to AI categorize all remaining
                st.session_state.user_labeled_examples = [] # No examples if skipped
                # combine labeling and remaining for ai_categorize_all stage
                st.session_state.all_debit_transactions_for_review = st.session_state.debit_transactions_for_labeling + st.session_state.remaining_debit_transactions
                st.session_state.debit_transactions_for_labeling = []
                st.session_state.remaining_debit_transactions = []
                st.rerun()
        if st.button("Upload a Different PDF", key="upload_diff_pdf_labeling_main_v2"):
            st.session_state.app_stage = "upload"; # Full reset done by upload stage start
            st.rerun()
        st.stop()

    st.write("Help the AI learn! Categorize these sample debit transactions accurately. Your labels will guide the AI for the rest.")
    current_labels_for_form = []
    with st.form(key="labeling_form_main_ui_v2"):
        # ... (form content for labeling - same as before) ...
        st.subheader("Label these debit transactions:")
        for i, t_dict_label in enumerate(st.session_state.debit_transactions_for_labeling):
            desc_lab = t_dict_label.get('description','N/A'); amt_lab = t_dict_label.get('parsed_amount',0.0); dt_lab = t_dict_label.get('date','')
            disp_txt_lab = f"**{desc_lab}** ({CURRENCY_SYMBOL}{amt_lab:.2f})" + (f" on {dt_lab}" if dt_lab else "")
            st.markdown(disp_txt_lab)
            default_category_labeling = COMMON_CATEGORIES_FOR_LABELLING[-2]
            default_idx_labeling = COMMON_CATEGORIES_FOR_LABELLING.index(default_category_labeling)
            chosen_category_labeling = st.selectbox(f"Category {i+1}", COMMON_CATEGORIES_FOR_LABELLING, index=default_idx_labeling, key=f"cat_lab_form_{t_dict_label['id']}_ui_v2", label_visibility="collapsed")
            if chosen_category_labeling != "Uncategorized":
                current_labels_for_form.append((t_dict_label, chosen_category_labeling))

        submit_labels_button_form = st.form_submit_button("Confirm Labels & Proceed to AI Categorization", type="primary")

    if submit_labels_button_form:
        st.session_state.user_labeled_examples = current_labels_for_form
        st.success(f"Labels submitted for {len(st.session_state.user_labeled_examples)} transactions.")
        # Prepare all_debit_transactions_for_review: user labeled ones + AI will categorize remaining
        # User labeled examples are already categorized.
        # AI will categorize st.session_state.remaining_debit_transactions
        st.session_state.app_stage = "ai_categorize_all" # New intermediate stage
        st.rerun()
    if st.button("Back to Upload (Discard Process)", key="b2u_label_main_ui_v2"):
        st.session_state.app_stage = "upload"; # Full reset done by upload stage start
        st.rerun()


# --- STAGE: AI Categorizes ALL (or remaining) transactions, then user reviews ---
elif st.session_state.app_stage == "ai_categorize_all":
    st.header("Step 3: AI Processing & Category Review")
    if not CLIENT_INSTANCE: st.error("Groq client N/A."); st.stop()

    # Transactions to be categorized by AI are those in 'remaining_debit_transactions'
    # OR all debit transactions if labeling was skipped.
    
    transactions_needing_ai_category = []
    if st.session_state.user_labeled_examples: # If user labeled some
        transactions_needing_ai_category = st.session_state.remaining_debit_transactions
    else: # If labeling was skipped, all debits need AI categorization
        transactions_needing_ai_category = st.session_state.all_debit_transactions_for_review # This should hold all debits if labeling skipped

    if transactions_needing_ai_category:
        with st.spinner(f"🤖 AI is categorizing {len(transactions_needing_ai_category)} debit transactions based on your examples (if any)..."):
            if IS_DEBUG_MODE:
                print(f"DEBUG CONSOLE: Sending {len(transactions_needing_ai_category)} transactions to AI for categorization (ai_categorize_all stage).")
                print(f"DEBUG CONSOLE: User examples being sent: {len(st.session_state.user_labeled_examples)}")
            
            ai_categorized_results_map = get_debit_categorization_with_llm(
                st.session_state.user_labeled_examples,
                transactions_needing_ai_category,
                COMMON_CATEGORIES_FOR_LABELLING
            )
            # Now, build the all_debit_transactions_for_review list
            # This list will store dicts like: {'transaction_obj': original_tx_dict, 'current_category': category_str, 'source': 'User'/'AI'}
            
            temp_review_list = []
            processed_ids_for_review_build = set()

            # Add user-labeled examples
            for tx_obj, cat_str in st.session_state.user_labeled_examples:
                tx_id = tx_obj['id']
                temp_review_list.append({'transaction_obj': tx_obj, 'current_category': cat_str, 'source': 'User Labeled Sample'})
                processed_ids_for_review_build.add(tx_id)
            
            # Add AI categorized transactions
            for tx_obj in transactions_needing_ai_category:
                tx_id = tx_obj['id']
                if tx_id not in processed_ids_for_review_build: # Should always be true if logic is correct
                    ai_cat = ai_categorized_results_map.get(tx_id, "Uncategorized (AI System Error)")
                    temp_review_list.append({'transaction_obj': tx_obj, 'current_category': ai_cat, 'source': 'AI Suggested'})
            
            st.session_state.all_debit_transactions_for_review = temp_review_list
            st.success("AI has processed transactions. Please review all categories below.")

    elif st.session_state.user_labeled_examples and not transactions_needing_ai_category : # User labeled all, nothing remaining for AI
        st.info("All selected debit transactions were labeled by you. Review them below.")
        temp_review_list = []
        for tx_obj, cat_str in st.session_state.user_labeled_examples:
             temp_review_list.append({'transaction_obj': tx_obj, 'current_category': cat_str, 'source': 'User Labeled Sample'})
        st.session_state.all_debit_transactions_for_review = temp_review_list
    else: # No user labels and no remaining debits somehow
        st.warning("No debit transactions to review or categorize.")
        if st.button("Start Over", key="st_over_aicatall"): st.session_state.app_stage = "upload"; st.rerun()
        st.stop()

    st.session_state.app_stage = "review_all" # Move to the new review stage
    st.rerun()


# --- STAGE: User Reviews ALL Categories ---
elif st.session_state.app_stage == "review_all":
    st.header("Step 4: Review All Debit Transaction Categories")
    if not CLIENT_INSTANCE: st.error("Groq client N/A."); st.stop()
    if not st.session_state.get('all_debit_transactions_for_review'):
        st.warning("No transactions available for review."); 
        if st.button("Start Over", key="st_over_review"): st.session_state.app_stage = "upload"; st.rerun()
        st.stop()

    st.markdown("Review the categories assigned by you or the AI. Make any corrections needed.")
    
    # Create a temporary list of categories for the selectboxes in the form
    # This list will be updated if the user changes a category
    # We need to ensure each selectbox reflects the current choice for that transaction
    
    # For the form, we need to ensure changes persist across interactions within the form if not submitted
    # Using st.data_editor might be complex for selectbox per row directly.
    # A loop of st.columns and selectboxes is feasible.

    transactions_to_display_for_review = st.session_state.all_debit_transactions_for_review

    # This form will handle edits for all transactions
    with st.form(key="review_all_form"):
        header_cols_review = st.columns([0.7, 2.5, 0.8, 1.2, 2]) # Date, Desc, Amt, Source, Category
        headers_review = ["Date", "Description", "Amount", "Categorized By", "Final Category"]
        for col, header in zip(header_cols_review, headers_review): col.markdown(f"**{header}**")
        st.markdown("---", unsafe_allow_html=True)

        new_categories_for_review = {} # To store changes made in this form submission

        for i, item_for_review in enumerate(transactions_to_display_for_review):
            tx_obj = item_for_review['transaction_obj']
            initial_category = item_for_review['current_category']
            source = item_for_review['source']
            tx_id = tx_obj['id'] # Use the stable ID

            row_cols_review = st.columns([0.7, 2.5, 0.8, 1.2, 2])
            row_cols_review[0].markdown(tx_obj.get('date', 'N/A'))
            with row_cols_review[1]:
                 st.markdown(f"<div title=\"{tx_obj['description']}\">{tx_obj.get('description','N/A')[:35]}{'...' if len(tx_obj.get('description','N/A')) > 35 else ''}</div>", unsafe_allow_html=True)
            row_cols_review[2].markdown(f"<span style='color:red;'>{CURRENCY_SYMBOL}{tx_obj.get('parsed_amount',0.0):.2f}</span>", unsafe_allow_html=True)
            row_cols_review[3].markdown(source)

            # Ensure category is valid for selectbox
            current_cat_for_select = initial_category
            if current_cat_for_select not in COMMON_CATEGORIES_FOR_LABELLING:
                current_cat_for_select = "Uncategorized" # Default if somehow invalid

            try:
                cat_select_idx_review = COMMON_CATEGORIES_FOR_LABELLING.index(current_cat_for_select)
            except ValueError:
                cat_select_idx_review = COMMON_CATEGORIES_FOR_LABELLING.index("Uncategorized")

            chosen_category_review = row_cols_review[4].selectbox(
                label=f"cat_review_{tx_id}", # Unique label
                options=COMMON_CATEGORIES_FOR_LABELLING,
                index=cat_select_idx_review,
                key=f"final_cat_select_{tx_id}", # Unique key for widget state
                label_visibility="collapsed"
            )
            new_categories_for_review[tx_id] = chosen_category_review # Store the choice from selectbox
            st.markdown("---", unsafe_allow_html=True) # Separator

        submit_review_button = st.form_submit_button("Confirm All Categories & Generate Overview", type="primary")

    if submit_review_button:
        # Update st.session_state.all_debit_transactions_for_review with the new_categories_for_review
        updated_review_list = []
        for item_in_review in st.session_state.all_debit_transactions_for_review:
            tx_obj_update = item_in_review['transaction_obj']
            tx_id_update = tx_obj_update['id']
            # Get the latest category chosen in the form for this transaction ID
            final_category_for_tx = new_categories_for_review.get(tx_id_update, item_in_review['current_category'])
            
            updated_review_list.append({
                'transaction_obj': tx_obj_update,
                'current_category': final_category_for_tx, # This is now the user-confirmed category
                'source': "User Reviewed" if item_in_review['current_category'] != final_category_for_tx else item_in_review['source'] # Update source if changed
            })
        st.session_state.all_debit_transactions_for_review = updated_review_list
        st.session_state.app_stage = "final_overview"
        st.rerun()

    if st.button("Back to Labeling Samples (If Applicable)", key="b2label_review"):
        # This would discard AI categorization of remaining and let user re-label samples
        st.session_state.app_stage = "labeling"
        # Keep user_labeled_examples as they were before AI cat, or reset them if desired
        # For now, just go back, assuming user wants to redo from labeling.
        # This might require resetting remaining_debit_transactions and all_debit_transactions_for_review
        st.session_state.remaining_debit_transactions = st.session_state.debit_transactions_for_labeling + st.session_state.remaining_debit_transactions # Re-pool all debits
        st.session_state.debit_transactions_for_labeling = [] # Will be re-sampled
        st.session_state.user_labeled_examples = []
        st.session_state.all_debit_transactions_for_review = []
        if 'categorized_results' in st.session_state: del st.session_state.categorized_results

        st.rerun()


# --- STAGE: Final Overview & Pie Chart ---
elif st.session_state.app_stage == "final_overview":
    st.header("Step 5: Final Financial Overview")
    if not CLIENT_INSTANCE: st.error("Groq client N/A."); st.stop()

    final_tx_for_summary = st.session_state.get('all_debit_transactions_for_review', [])
    if not final_tx_for_summary:
        st.error("No finalized categorized debit transactions available for overview.")
        if st.button("Start Over", key="st_over_final"): st.session_state.app_stage = "upload"; st.rerun()
        st.stop()

    final_category_totals_overview = defaultdict(float)
    # Rebuild the display list for the dataframe to show only relevant columns for final overview
    display_list_overview = []

    for item in final_tx_for_summary:
        tx_obj = item['transaction_obj']
        category = item['current_category'] # This is the user-confirmed or final AI category
        amount = tx_obj.get('parsed_amount', 0.0)
        
        display_list_overview.append({
            "Date": tx_obj.get('date', 'N/A'),
            "Description": tx_obj.get('description', 'N/A'),
            "Amount": amount,
            "Category": category,
            "Source": item.get('source', 'N/A') # Could show if it was user-labeled, AI, or user-reviewed AI
        })
        if not category.startswith("Uncategorized") and category != "Error Categorizing" and not category.startswith("Error -"):
            final_category_totals_overview[category] += amount
            
    st.success("Displaying Final Categorized Debit Transactions!")
    try: 
        display_list_overview.sort(key=lambda x: str(x.get('Date','9999-99-99') or '9999-99-99'))
    except Exception as final_sort_e_overview: st.warning(f"Could not sort transactions by date: {final_sort_e_overview}")
    
    st.dataframe(
        display_list_overview, 
        use_container_width=True, 
        column_config={
            "Amount": st.column_config.NumberColumn(f"Amount ({CURRENCY_SYMBOL})", format=f"{CURRENCY_SYMBOL}%.2f"), 
            "Description": "Description", "Category": "Category", "Source": "Categorization Source"
        }
    )
    
    st.subheader(f"Final Debit Spend Distribution ({CURRENCY_SYMBOL})")
    if final_category_totals_overview:
        pie_chart_figure_final = create_pie_chart(final_category_totals_overview)
        if pie_chart_figure_final: 
            st.pyplot(pie_chart_figure_final)
        # else: create_pie_chart prints message
    elif IS_DEBUG_MODE: st.warning("Debug: Could not aggregate spending for final pie chart.")
    
    st.subheader("💡 AI Financial Suggestions")
    if final_category_totals_overview: 
        if 'suggestions_generated_flag' not in st.session_state: st.session_state.suggestions_generated_flag = False
        
        get_suggestions_button_disabled_final = st.session_state.suggestions_generated_flag or not CLIENT_INSTANCE
        if st.button("Get AI Financial Suggestions", key="get_suggestions_final_btn",type="primary",disabled=get_suggestions_button_disabled_final):
            with st.spinner("🤖 Generating final suggestions..."):
                suggestions_text_final = get_llm_suggestions(dict(sorted(final_category_totals_overview.items(),key=lambda item_val_sugg: item_val_sugg[1],reverse=True)))
                st.session_state.ai_suggestions_text_output = suggestions_text_final
                st.session_state.suggestions_generated_flag = True; st.rerun()
        
        if st.session_state.get('ai_suggestions_text_output'):
             st.markdown(st.session_state.ai_suggestions_text_output)
    elif IS_DEBUG_MODE: st.info("Debug: No expense data for final suggestions.")

    if st.button("Start Over with a New PDF", key="start_over_very_final_btn"):
        st.session_state.app_stage = "upload"
        for key_to_reset_final_btn in default_state.keys(): 
            st.session_state[key_to_reset_final_btn] = default_state[key_to_reset_final_btn] if key_to_reset_final_btn != 'app_stage' else 'upload'
        for k_del_sugg_final in ['ai_suggestions_text_output','suggestions_generated_flag']: 
            if k_del_sugg_final in st.session_state: del st.session_state[k_del_sugg_final]
        st.rerun()


# --- Fallback for initial load or if app_stage is somehow not set ---
elif not st.session_state.app_stage: 
    st.session_state.app_stage = "upload"; st.rerun()

# --- Sidebar Info ---
# ... (Sidebar content - same as before) ...
st.sidebar.header("About"); st.sidebar.info("AI (Groq Llama 3) to extract & categorize DEBIT transactions from PDF bank statements.")
st.sidebar.markdown("---"); st.sidebar.header("How It Works"); st.sidebar.markdown(f"""1. Upload PDF\n2. AI Extract ({GROQ_MODEL_EXTRACTION})\n3. Label Samples\n4. Review AI & User Categories\n5. View Overview & Get Insights""")
st.sidebar.markdown("---"); st.sidebar.header("Notes"); st.sidebar.warning("- Accuracy varies by PDF.\n- Groq API limits apply.\n- API Key via secrets.")
