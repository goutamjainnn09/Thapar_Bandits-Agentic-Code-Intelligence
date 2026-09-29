import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType
MODEL_NAME="BAAI/bge-small-en-v1.5"
TRUST_REMOTE_CODE = False
MAX_SEQ_LENGTH=512
BATCH_SIZE = 32
USE_IDENTIFIER_SPLIT = False

def split_identifiers(text: str)->str:
    """preProcessInput -> pre Process Input, my_var_name -> my var name."""
    text = re.sub(r"([a-z0-9])([A-Z])",r"\1 \2",text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])",r"\1 \2",text)
    return text.replace("_"," ")

class PrePostPipelineEncoder(AbsEncoder):
    def __init__(self,model_name: str = MODEL_NAME):
        self.model=SentenceTransformer(model_name, device = "cpu",trust_remote_code=TRUST_REMOTE_CODE)
        self.model.max_seq_length=MAX_SEQ_LENGTH
        
        self.short_name = model_name.split("/")[-1]
        if USE_IDENTIFIER_SPLIT:
            self.short_name += "-split"

        self.mteb_model_meta= ModelMeta(
            loader = None,
            name = "prism-theme1/{self.short_name}",
            revision="v0.1",
            release_date="2026-09-15",
            embed_dim=self.model.get_sentence_embedding_dimension(),
            similarity_fn_name=ScoringFunction.COSINE,
            languages=["eng-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=512,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["Sentence Transformers"],
            use_instructions=False,
            training_datasets=None,
        )
    def _preprocess_query(self,texts: list[str])->list[str]:
        """Natural-lamguage side (the question)."""
        return texts
    def _preprocess_document(self,texts: list[str])->list[str]:
        """Code side (the snippets)."""
        if USE_IDENTIFIER_SPLIT:
            return [split_identifiers(t) for t in texts]
        return texts
    def encode(self,inputs,*,task_metadata,hf_split: str,hf_subset: str,prompt_type=None,**kwargs)->np.ndarray:
        texts: list[str]=[]
        for batch in inputs:
            texts.extend(batch["text"])
        is_query = prompt_type==PromptType.query
        texts = (
            self._preprocess_query(texts)
            if is_query
            else self._preprocess_document(texts)
        )
        return self.model.encode(
            texts,
            batch_size=kwargs.get("batch_size",BATCH_SIZE),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
def main()-> None:
    model=PrePostPipelineEncoder()
    task = mteb.get_task("AppsRetrieval")
    result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": BATCH_SIZE})
    data = list(result.task_results)[0].to_dict()

    for path in ("appsretrieval_results.json",f"results_{model.short_name}.json"):
        with open(path,"w") as f:
            json.dump(data,f,indent=2,default=str)
    try:
        scores = data["scores"]["test"][0]
        print("NDCG@10:", scores["ndcg_at_10"])
        print("MRR@10 :", scores["mrr_at_10"])
    except Exception as e:
        print("Results saved, but could not print the scores:", e)
if __name__=="__main__":
    main()