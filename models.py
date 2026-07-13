"""
Load and manage ML models
"""

import os
import json
import torch
from sentence_transformers import SentenceTransformer, SparseEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from json_repair import repair_json

import config


def _materialize_meta_to(device="cpu", dtype=torch.float32):
    """Context manager that patches Module.to() to materialize meta tensors before moving."""
    _orig_to = torch.nn.Module.to

    def _patched_to(self, *args, **kwargs):
        for module in self.modules():
            for name, param in list(module.named_parameters(recurse=False)):
                if param.is_meta:
                    real = torch.empty(param.shape, device="cpu", dtype=dtype)
                    torch.nn.init.xavier_uniform_(real)
                    setattr(module, name, torch.nn.Parameter(real))
            for name, buf in list(module.named_buffers(recurse=False)):
                if buf.is_meta:
                    real = torch.zeros(buf.shape, device="cpu", dtype=buf.dtype if buf.dtype != torch.meta else dtype)
                    setattr(module, name, real)
        return _orig_to(self, *args, **kwargs)

    torch.nn.Module.to = _patched_to
    return _orig_to


class DenseEmbedder:
    """
    Dense vector embedding using Harrier model
    Output: 640-dimensional vector
    """
    def __init__(self):
        print("Loading dense model:", config.DENSE_MODEL)
        self.model = SentenceTransformer(config.DENSE_MODEL)
        self.model.max_seq_length = 8192
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, batch_size=config.BATCH_SIZE, show_progress_bar=False)


class SparseEmbedder:
    """
    Sparse vector embedding using Arabic SPLADE via SparseEncoder
    Output: dict of {token_name: weight} (ES sparse_vector format)
    """
    def __init__(self):
        print("Loading sparse model:", config.SPARSE_MODEL)
        self.model = self._safe_load()
        try:
            self.tokenizer = self.model.tokenizer
        except:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.SPARSE_MODEL)

    @staticmethod
    def _safe_load():
        orig = _materialize_meta_to()
        try:
            model = SparseEncoder(config.SPARSE_MODEL)
            model = model.float().to("cpu")
            return model
        finally:
            torch.nn.Module.to = orig
    
    def encode(self, texts):
        """
        Encode texts to sparse vectors
        Returns: list of dict {token_name: weight}
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts)
        
        results = []
        for emb in embeddings:
            decoded_dict = self._tensor_to_dict(emb)
            results.append(decoded_dict)
        
        return results
    
    def _tensor_to_dict(self, emb):
        """Convert sparse tensor or object to dict with token names"""
        import torch
        
        # Handle PyTorch sparse tensor
        if hasattr(emb, 'indices') and hasattr(emb, 'values'):
            # Coalesce the tensor first (required for sparse tensors)
            if hasattr(emb, 'coalesce'):
                emb = emb.coalesce()
            
            indices = emb.indices().squeeze().tolist()
            values = emb.values().tolist()
            
            if not isinstance(indices, list):
                indices = [indices]
                values = [values]
            
            sparse_dict = {}
            for idx, val in zip(indices, values):
                if val > 0.01:
                    # Decode token ID to token name
                    try:
                        token_name = self.tokenizer.decode([int(idx)]).strip()
                        # Filter out invalid tokens
                        if self._is_valid_token(token_name):
                            sparse_dict[token_name] = float(val)
                    except:
                        pass
            return sparse_dict
        
        # Handle dict-like object
        if hasattr(emb, 'to_dict'):
            sparse_dict = emb.to_dict()
        elif hasattr(emb, 'keys'):
            sparse_dict = dict(emb)
        elif isinstance(emb, dict):
            sparse_dict = emb
        else:
            return {}
        
        # Decode token IDs to names
        return self._decode_tokens(sparse_dict)
    
    def _is_valid_token(self, token):
        """Check if token is valid for ES sparse_vector"""
        if not token or len(token) == 0:
            return False
        
        # Skip BERT wordpiece tokens (##xxx)
        if token.startswith('##'):
            return False
        
        # Skip sentencepiece tokens (▁xxx)
        if token.startswith('▁'):
            token = token[1:]  # Remove prefix and check rest
            if not token:
                return False
        
        # Skip tokens with only special chars
        if all(not c.isalnum() for c in token):
            return False
        
        return True
    
    def _decode_tokens(self, sparse_dict):
        """Convert token IDs to token names for ES sparse_vector format"""
        decoded = {}
        for token_key, weight in sparse_dict.items():
            if weight <= 0.001:
                continue
            
            token_key_str = str(token_key)
            
            # If it's a numeric token ID, decode it
            if token_key_str.isdigit():
                try:
                    token_id = int(token_key_str)
                    token_name = self.tokenizer.decode([token_id]).strip()
                    if self._is_valid_token(token_name):
                        decoded[token_name] = float(weight)
                except:
                    pass
            else:
                # Already a token name - check if valid
                if self._is_valid_token(token_key_str):
                    decoded[token_key_str] = float(weight)
        
        return decoded
    
    def encode_query(self, texts):
        """Encode queries - returns decoded token names"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode_query(texts)
        
        results = []
        for emb in embeddings:
            decoded_dict = self._tensor_to_dict(emb)
            results.append(decoded_dict)
        
        return results
    
    def encode_document(self, texts):
        """Encode documents"""
        if isinstance(texts, str):
            texts = [texts]
        return self.encode(texts)


class TextCorrector:
    """
    Correct Arabic text spelling/grammar errors using T5
    Runs in separate queue (non-blocking)
    """
    def __init__(self):
        print("Loading text correction model:", config.TEXT_CORRECTION_MODEL)
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        self.device = "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(config.TEXT_CORRECTION_MODEL)
        self.model = T5ForConditionalGeneration.from_pretrained(config.TEXT_CORRECTION_MODEL)
        self.model = self.model.to(self.device)
        self.model.eval()
        torch.set_num_threads(os.cpu_count())
        print("[TextCorrector] Model loaded (CPU mode)")
    
    def correct(self, text: str, max_length: int = 512) -> str:
        """Correct text spelling/grammar"""
        if not text or len(text.strip()) < 5:
            return text
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected if corrected else text
        except Exception as e:
            print(f"[TextCorrector] Error: {e}")
            return text


class MetadataExtractor:
    """
    Extract metadata using Qwen LLM with LoRA adapter
    Output: {story_titel, story_keywords, story_sammary, story_category, story_entities}
    """
    def __init__(self):
        self.device = "cpu"
        
        if config.QWEN_ADAPTER and os.path.exists(config.QWEN_ADAPTER):
            print("Loading tokenizer from adapter:", config.QWEN_ADAPTER)
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.QWEN_ADAPTER,
                local_files_only=True,
                trust_remote_code=True
            )
        else:
            print("Loading tokenizer from:", config.QWEN_MODEL)
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.QWEN_MODEL,
                trust_remote_code=True
            )
        
        print("Loading base model:", config.QWEN_MODEL)
        orig_to = _materialize_meta_to()
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.QWEN_MODEL,
                torch_dtype=torch.float32,
                device_map={"": self.device},
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        finally:
            torch.nn.Module.to = orig_to
        
        if config.QWEN_ADAPTER and os.path.exists(config.QWEN_ADAPTER):
            print("Loading LoRA adapter from:", config.QWEN_ADAPTER)
            orig_to = _materialize_meta_to()
            try:
                peft_model = PeftModel.from_pretrained(base_model, config.QWEN_ADAPTER)
            finally:
                torch.nn.Module.to = orig_to
            self.model = peft_model.float().to(self.device)
        else:
            print("No adapter found, using base model")
            self.model = base_model
        
        inner = self.model
        if hasattr(inner, 'base_model'):
            inner = inner.base_model.model
        if hasattr(inner, 'lm_head') and hasattr(inner, 'model'):
            embed = getattr(inner.model, 'embed_tokens', None) or getattr(inner.model, 'wte', None)
            if embed is not None and inner.lm_head.weight.shape == embed.weight.shape:
                inner.lm_head.weight = embed.weight
                print("[MetadataExtractor] Tied lm_head to embed_tokens")
        
        self.model.eval()
        torch.set_num_threads(os.cpu_count())
        
        self._build_schema()
    
    def _build_schema(self):
        """Build extraction schema with configurable summary length"""
        min_summary = config.SUMMARY_MIN_POINTS
        max_summary = config.SUMMARY_MAX_POINTS
        
        schema = {
            "properties": {
                "story_titel": {
                    "type": "string",
                    "minLength": 10,
                    "maxLength": 100,
                    "description": "A fully informative and SEO optimized title of the story."
                },
                "story_keywords": {
                    "type": "array",
                    "minItems": 2,
                    "items": {"type": "string"},
                    "description": "Relevant Keywords associated with the story."
                },
                "story_sammary": {
                    "type": "array",
                    "minItems": min_summary,
                    "maxItems": max_summary,
                    "items": {"type": "string"},
                    "description": f"Summarized key points about the story ({min_summary}-{max_summary} points)"
                },
                "story_category": {
                    "type": "string",
                    "enum": ["politics", "sports", "art", "technology", "economy", 
                             "health", "entertainment", "science", "religion", "not_specified"],
                    "description": "category of the news story."
                },
                "story_entities": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_value": {"type": "string"},
                            "entity_type": {
                                "type": "string",
                                "enum": ["person-male", "person-female", "location", 
                                        "organization", "event", "time", "quantity", 
                                        "money", "product", "law", "disease", 
                                        "artifact", "not_specified"]
                            }
                        }
                    },
                    "description": "List of identified entities in the story."
                }
            },
            "required": ["story_titel", "story_keywords", "story_sammary", 
                        "story_category", "story_entities"]
        }
        self.extraction_schema = json.dumps(schema, ensure_ascii=False)
        
        self.system_prompt = "\n".join([
            "You are a professional NLP data parser.",
            "Follow the provided `Task` by the user and the `Output Scheme` to generate the `Output JSON`.",
            "Do not generate any introduction or conclusion."
        ])
    
    def _build_prompt(self, story: str) -> str:
        """Build extraction prompt using chat template"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n".join([
                "# Story:", story.strip(), "",
                "# Task:", "Extract the story details into a JSON.", "",
                "# Output Scheme:", self.extraction_schema, "",
                "# Output JSON:", "```json",
            ])},
        ]
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response from model - ONLY decode new tokens"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # FIX: Only decode NEW tokens (slice off input tokens)
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def _extract_json(self, response: str) -> dict:
        """Extract and repair JSON from response - matches notebook pattern"""
        try:
            # Find JSON object boundaries
            start = response.find("{")
            end = response.rfind("}") + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                result = repair_json(json_str)
                
                # repair_json may return string or dict
                if isinstance(result, str):
                    result = json.loads(result)
                
                if isinstance(result, dict):
                    return self._fix_entities(result)
            
            return {"error": "No JSON object found", "raw_response": response}
            
        except Exception as e:
            return {"error": str(e), "raw_response": response}
    
    def _fix_entities(self, data: dict) -> dict:
        """Fix entities that are strings instead of dicts"""
        if not isinstance(data, dict):
            return data
        
        entities = data.get("story_entities", [])
        if isinstance(entities, list):
            fixed_entities = []
            for e in entities:
                if isinstance(e, str):
                    try:
                        fixed_e = json.loads(e)
                        fixed_entities.append(fixed_e)
                    except:
                        if "{" in e:
                            try:
                                fixed_e = repair_json(e)
                                fixed_entities.append(fixed_e)
                            except:
                                pass
                elif isinstance(e, dict):
                    fixed_entities.append(e)
            data["story_entities"] = fixed_entities
        
        return data
    
    def extract(self, text: str) -> dict:
        """
        Extract metadata from Arabic text
        Returns: dict with story_titel, story_keywords, story_sammary, story_category, story_entities
        """
        prompt = self._build_prompt(text[:3000])
        response = self._generate(prompt, max_new_tokens=512)
        result = self._extract_json(response)
        
        if "error" in result:
            return {
                "story_titel": "",
                "story_keywords": [],
                "story_sammary": [],
                "story_category": "not_specified",
                "story_entities": []
            }
        
        return result


_dense_model = None
_sparse_model = None
_qwen_model = None
_text_corrector = None


def get_dense_model():
    global _dense_model
    if _dense_model is None:
        _dense_model = DenseEmbedder()
    return _dense_model


def get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseEmbedder()
    return _sparse_model


def get_qwen_model():
    global _qwen_model
    if _qwen_model is None:
        _qwen_model = MetadataExtractor()
    return _qwen_model


def get_text_corrector():
    global _text_corrector
    if _text_corrector is None:
        _text_corrector = TextCorrector()
    return _text_corrector
