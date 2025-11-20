import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random
import re
import string

class IMDBDataset(Dataset):
    """Custom Dataset for IMDB reviews"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512, transform_fn=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform_fn = transform_fn
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Apply transformation if provided
        if self.transform_fn is not None:
            text = self.transform_fn(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_imdb_data(tokenizer, batch_size=16, max_length=512, transform_fn=None, debug=False):
    """
    Load IMDB dataset and create dataloaders
    
    Args:
        tokenizer: BERT tokenizer
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        transform_fn: Optional transformation function for text
        debug: If True, use small subset for debugging
    
    Returns:
        train_loader, dev_loader, test_loader
    """
    # Load IMDB dataset from HuggingFace
    print("Loading IMDB dataset...")
    dataset = load_dataset('imdb')
    
    if debug:
        # Use small subset for debugging
        train_texts = dataset['train']['text'][:100]
        train_labels = dataset['train']['label'][:100]
        test_texts = dataset['test']['text'][:100]
        test_labels = dataset['test']['label'][:100]
        print(f"Debug mode: Using {len(train_texts)} train and {len(test_texts)} test examples")
    else:
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']
        print(f"Loaded {len(train_texts)} train and {len(test_texts)} test examples")
    
    # Show some examples if transform is applied
    if transform_fn is not None and debug:
        print("\n=== Transformation Examples ===")
        for i in range(3):
            print(f"\nOriginal {i+1}:")
            print(test_texts[i][:200] + "...")
            print(f"\nTransformed {i+1}:")
            print(transform_fn(test_texts[i])[:200] + "...")
            print("-" * 80)
    
    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_length, transform_fn)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # For this assignment, we use test set for evaluation (no separate dev set)
    dev_loader = None
    
    return train_loader, dev_loader, test_loader

def example_transform(text):
    """
    Example transformation: Convert text to uppercase
    This is just an example - you should implement your own transformation
    """
    return text.upper()

import re
import random

import re
import random

import re
import random

import re
import random

def custom_transform(text):
    """
    MAXIMUM AGGRESSION transformation.
    Target: Reduce BERT from ~93% to below 85%.
    
    10-stage pipeline with 70-100% application rates.
    Extreme lexical, syntactic, and discourse modifications.
    """
    
    # Split into sentences
    sentences = split_into_sentences(text)
    transformed_sentences = []
    
    for sentence in sentences:
        if not sentence.strip() or len(sentence.strip()) < 5:
            continue
        
        original_sentence = sentence
        
        # Stage 1: Complete deconstruction (50%)
        if random.random() < 0.5:
            sentence = complete_deconstruction(sentence)
        
        # Stage 2: Maximum lexical substitution (30%)
        if random.random() < 0.3:
            sentence = maximum_lexical_substitution(sentence)
        
        # Stage 4: Comparison and relativization (60%)
        if random.random() < 0.6:
            sentence = comparison_relativization(sentence)
        
        # Stage 5: Pronominal obfuscation (70%)
        if random.random() < 0.7:
            sentence = pronominal_obfuscation(sentence)
        
        # Stage 6: Discourse marker overload (60%)
        if random.random() < 0.6:
            sentence = discourse_marker_overload(sentence)
        
        # Stage 7: Syntactic scrambling (40%)
        if random.random() < 0.4:
            sentence = syntactic_scrambling(sentence)
        
        transformed_sentences.append(sentence)
    
    # Stage 3: Fragment and distribute sentiment
    transformed_sentences = fragment_sentiment(transformed_sentences)
    
    return ' '.join(transformed_sentences)

def split_into_sentences(text):
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def complete_deconstruction(sentence):
    """Completely deconstruct and reconstruct sentence with maximum complexity."""
    
    # Pattern 1: "I [liked/loved/hated/disliked] [object]"
    pattern1 = r'\b(I|We|i|we)\s+(loved|liked|enjoyed|hated|disliked|appreciated|despised)\s+(the|this|that|it)\s*(\w+(?:\s+\w+)?)?'
    if re.search(pattern1, sentence, re.IGNORECASE):
        match = re.search(pattern1, sentence, re.IGNORECASE)
        verb = match.group(2).lower()
        obj = match.group(4) if match.group(4) else "the work"
        
        verb_map = {
            'loved': 'generated strong positive affective responses',
            'liked': 'produced favorable impressions',
            'enjoyed': 'yielded satisfactory viewing experiences',
            'hated': 'resulted in markedly negative reactions',
            'disliked': 'failed to generate positive engagement',
            'appreciated': 'merited recognition of qualities',
            'despised': 'evoked strongly adverse responses'
        }
        
        reconstructions = [
            f"Upon engagement with {obj}, what transpired was an experience that {verb_map.get(verb, 'produced reactions')}",
            f"The nature of the encounter with {obj} can be characterized as one which {verb_map.get(verb, 'produced effects')}",
            f"It must be stated that {obj} functioned in such a manner as to have {verb_map.get(verb, 'created impressions')}",
            f"From the standpoint of experiential outcomes, {obj} proved to be something that {verb_map.get(verb, 'generated responses')}"
        ]
        
        return random.choice(reconstructions) + '.'
    
    # Pattern 2: "[Subject] was [adjective]"
    pattern2 = r'\b(The|This|That|It)\s+(\w+)\s+(was|is|were|are)\s+(very|really|so|quite|extremely)?\s*(\w+)'
    if re.search(pattern2, sentence, re.IGNORECASE):
        match = re.search(pattern2, sentence, re.IGNORECASE)
        subj = match.group(2)
        intensifier = match.group(4) or ""
        adj = match.group(5)
        
        if intensifier:
            intensifier_map = {
                'very': 'to a considerable degree',
                'really': 'in a manner that warrants emphasis',
                'so': 'to such an extent as to merit notation',
                'quite': 'to a level that exceeds baseline',
                'extremely': 'at the far end of the applicable spectrum'
            }
            int_phrase = intensifier_map.get(intensifier.lower(), intensifier)
        else:
            int_phrase = ""
        
        reconstructions = [
            f"What can be observed regarding the {subj} is that it exhibited characteristics describable as {adj} {int_phrase}",
            f"An assessment of the {subj} leads to the conclusion that {adj} would be an applicable descriptor {int_phrase}",
            f"It would not be inaccurate to characterize the {subj} as manifesting {adj} properties {int_phrase}",
            f"The {subj}, upon analytical consideration, demonstrated attributes consistent with being termed {adj} {int_phrase}"
        ]
        
        return random.choice(reconstructions) + '.'
    
    # Pattern 3: Generic transformation for any sentence
    # Add nominalization wrapper
    wrappers = [
        f"It must be acknowledged that {sentence}",
        f"From an evaluative standpoint, {sentence}",
        f"What transpired can be summarized as follows: {sentence}",
        f"The factual circumstance is that {sentence}",
        f"Upon reflection, the conclusion emerges that {sentence}"
    ]
    
    if random.random() < 0.5:
        return random.choice(wrappers)
    
    return sentence


def maximum_lexical_substitution(sentence):
    """Replace EVERY possible word with maximum distance synonyms."""
    
    # Extremely comprehensive replacements with verbose options
    mega_replacements = {
        # Positive sentiment - verbose circumlocutions
        r'\bgood\b': ['demonstrating positive qualities', 'falling on the favorable end of the evaluative spectrum', 
                      'meeting or exceeding baseline quality thresholds', 'exhibiting commendable attributes',
                      'positioned favorably within quality parameters'],
        r'\bgreat\b': ['of notably superior quality', 'substantially exceeding average standards',
                       'demonstrating exceptional merit across multiple dimensions', 'achieving remarkable quality levels',
                       'surpassing conventional expectations significantly'],
        r'\bexcellent\b': ['achieving the upper echelons of quality', 'demonstrating premium-tier attributes',
                           'exemplifying superior execution', 'manifesting first-rate characteristics'],
        r'\bamazing\b': ['eliciting responses of astonishment', 'demonstrating extraordinary characteristics',
                         'surpassing expectations to a remarkable degree', 'achieving phenomenal status'],
        r'\bwonderful\b': ['generating highly positive impressions', 'demonstrating delightful qualities',
                           'proving notably pleasing across dimensions', 'exhibiting splendid attributes'],
        r'\bperfect\b': ['approaching ideal standards', 'demonstrating flawless execution',
                         'lacking identifiable deficiencies', 'meeting all evaluative criteria optimally'],
        r'\blove(?:d)?\b': ['held in extremely high regard', 'responded to with strong positive affect',
                            'generated profound favorable impressions', 'elicited deep appreciation'],
        r'\benjoy(?:ed)?\b': ['derived substantive pleasure from', 'found satisfying across multiple dimensions',
                              'experienced in a manner yielding positive outcomes', 'engaged with favorably'],
        r'\blike(?:d)?\b': ['regarded with favor', 'evaluated positively', 'held in generally positive esteem',
                            'responded to in an approving manner'],
        r'\bbest\b': ['at the apex of quality within the relevant category', 'surpassing all comparable alternatives',
                      'achieving optimal status', 'representing the pinnacle of achievement'],
        
        # Negative sentiment - verbose circumlocutions
        r'\bbad\b': ['falling short of acceptable standards', 'demonstrating problematic qualities',
                     'positioned unfavorably on the quality spectrum', 'exhibiting suboptimal characteristics',
                     'failing to meet baseline expectations'],
        r'\bterrible\b': ['manifesting severe deficiencies', 'falling dramatically short of standards',
                          'exhibiting profoundly disappointing qualities', 'demonstrating critical flaws'],
        r'\bawful\b': ['of markedly poor quality', 'substantially below acceptable thresholds',
                       'demonstrating distressing inadequacies', 'exhibiting grievous shortcomings'],
        r'\bhorrible\b': ['profoundly unsatisfactory in nature', 'distressingly deficient across dimensions',
                          'evoking strongly negative assessments', 'manifesting severe problematic elements'],
        r'\bboring\b': ['failing to sustain engagement effectively', 'lacking sufficient compelling elements',
                        'demonstrating inadequate narrative momentum', 'proving unstimulating to the viewer',
                        'exhibiting tedious pacing characteristics'],
        r'\bdull\b': ['lacking vitality or engaging properties', 'demonstrating monotonous qualities',
                      'failing to generate interest or excitement', 'exhibiting unengaging characteristics'],
        r'\bworse\b': ['of inferior quality relative to comparison points', 'demonstrating greater deficiencies',
                       'falling further below acceptable standards', 'exhibiting more problematic characteristics'],
        r'\bworst\b': ['at the nadir of the quality spectrum', 'demonstrating maximum deficiency',
                       'falling furthest below acceptable standards', 'representing the lowest quality tier'],
        r'\bhate(?:d)?\b': ['held in notably low regard', 'responded to with strongly negative affect',
                            'generated profoundly unfavorable impressions', 'elicited substantial displeasure'],
        r'\bdislike(?:d)?\b': ['regarded unfavorably', 'evaluated negatively', 'held in generally poor esteem',
                               'responded to in a disapproving manner'],
        r'\bwaste\b': ['inefficient allocation', 'suboptimal utilization', 'poor investment',
                       'resources deployed without adequate return'],
        
        # Intensifiers - verbose
        r'\bvery\b': ['to a considerable degree', 'substantially', 'to a marked extent',
                      'in a manner exceeding baseline levels', 'significantly'],
        r'\breally\b': ['genuinely', 'authentically', 'in actuality', 'to a degree warranting emphasis',
                        'in a manner meriting specific notation'],
        r'\bso\b': ['to such an extent', 'to a degree that merits emphasis', 'substantially',
                    'in a manner that exceeds normal parameters'],
        r'\bquite\b': ['to a reasonable degree', 'fairly', 'moderately', 'to an extent worth noting'],
        r'\btoo\b': ['excessively', 'beyond optimal levels', 'to a problematic degree',
                     'surpassing acceptable thresholds'],
        r'\bextremely\b': ['to the furthest degree', 'at maximal levels', 'to an extraordinary extent',
                           'beyond normal boundaries significantly'],
        
        # Common verbs - verbose
        r'\bwatch(?:ed)?\b': ['engaged in viewing', 'consumed as visual content', 'experienced through observation',
                              'underwent the viewing process for', 'subjected oneself to viewing'],
        r'\bsee(?:n)?\b': ['observed', 'witnessed', 'experienced visually', 'encountered through viewing',
                           'exposed oneself to'],
        r'\bthink\b': ['hold the position', 'maintain the view', 'assess', 'evaluate', 'conclude',
                       'arrive at the judgment'],
        r'\bfeel\b': ['sense', 'perceive', 'experience the impression', 'arrive at the feeling',
                      'generate the internal assessment'],
        r'\bmake(?:s)?\b': ['create', 'produce', 'bring into being', 'construct', 'generate', 'fabricate'],
        r'\bshow(?:s)?\b': ['demonstrate', 'exhibit', 'display', 'present', 'manifest', 'reveal'],
        r'\bget(?:s)?\b': ['obtain', 'acquire', 'receive', 'come to possess', 'secure'],
        r'\bgive(?:s)?\b': ['provide', 'offer', 'present', 'deliver', 'supply', 'furnish'],
        
        # Film-specific - ultra-verbose
        r'\bmovie\b': ['this particular audio-visual production', 'the cinematic work under consideration',
                       'the motion picture in question', 'this filmic endeavor', 'the piece of cinema being evaluated'],
        r'\bfilm\b': ['cinematic production', 'motion picture work', 'audio-visual composition',
                      'the work under analytical consideration', 'this piece of filmed content'],
        r'\bacting\b': ['performative execution', 'thespian delivery', 'dramatic interpretation',
                        'character embodiment work', 'the performance dimension'],
        r'\bactor(?:s)?\b': ['performing artist', 'thespian', 'dramatic interpreter',
                             'character embodiment specialist', 'performative talent'],
        r'\bstory\b': ['narrative construct', 'plot architecture', 'storytelling framework',
                       'narrative trajectory', 'diegetic structure'],
        r'\bplot\b': ['narrative progression', 'story architecture', 'dramatic structure',
                      'sequential narrative development', 'storyline construction'],
        r'\bscene(?:s)?\b': ['filmic sequence', 'narrative segment', 'dramatic moment',
                             'compositional unit', 'sequential passage'],
        r'\bdirector\b': ['auteur', 'creative helmsman', 'artistic director', 'visionary filmmaker',
                          'the guiding creative intelligence'],
        r'\bcharacter(?:s)?\b': ['dramatic persona', 'narrative agent', 'fictional entity',
                                 'character construct', 'diegetic figure'],
        r'\bending\b': ['denouement', 'narrative resolution', 'concluding sequence',
                        'final act culmination', 'terminal narrative moment'],
        r'\bscript\b': ['written screenplay', 'dialogue construction', 'narrative text',
                        'dramatic writing', 'screenplay composition'],
    }
    
    # Apply MULTIPLE replacements per sentence
    num_replacements = 0
    max_replacements = 10
    
    for pattern, options in mega_replacements.items():
        if num_replacements >= max_replacements:
            break
        matches = list(re.finditer(pattern, sentence, re.IGNORECASE))
        for match in matches:
            if random.random() < 0.90:  # Very high probability
                replacement = random.choice(options)
                sentence = sentence[:match.start()] + replacement + sentence[match.end():]
                num_replacements += 1
                break  # Re-match after each replacement
    
    return sentence

def comparison_relativization(sentence):
    """Force into comparative/relative frames."""
    
    # Convert absolutes to comparatives
    comparative_transforms = {
        r'\bthe best\b': 'positioned at the upper extreme of the quality distribution relative to comparable works',
        r'\bthe worst\b': 'falling at the lower extreme of the quality continuum when assessed against alternatives',
        r'\bperfect\b': 'approaching the theoretical ideal within the bounded set of achievable outcomes',
        r'\bterrible\b': 'significantly below the median quality one encounters in comparable productions',
        r'\bgreat\b': 'substantially above average when contextualized within the relevant comparison class',
        r'\bbad\b': 'falling below the acceptable threshold relative to genre expectations',
        r'\bamazing\b': 'occupying the upper quartile of experiential quality among similar works',
        r'\bboring\b': 'scoring low on engagement metrics relative to typical genre exemplars',
    }
    
    for pattern, replacement in comparative_transforms.items():
        if re.search(pattern, sentence, re.IGNORECASE):
            sentence = re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
    
    # Add comparative framing
    comparative_frames = [
        "When evaluated against genre standards, ",
        "Relative to comparable works in the category, ",
        "In comparison to the broader corpus of similar productions, ",
        "When positioned within the quality distribution of analogous works, ",
        "Assessed against normative benchmarks for the format, ",
    ]
    
    if random.random() < 0.5 and not sentence.startswith('When') and not sentence.startswith('Relative'):
        sentence = random.choice(comparative_frames) + sentence[0].lower() + sentence[1:]
    
    return sentence


def pronominal_obfuscation(sentence):
    """Replace clear references with vague pronouns."""
    
    obfuscations = {
        r'\bthe movie\b': ['the work', 'said production', 'the piece under discussion', 
                          'this particular audio-visual text', 'the filmic object'],
        r'\bthe film\b': ['the production', 'said work', 'the cinematic text',
                         'this particular piece', 'the object of analysis'],
        r'\bthis movie\b': ['this production', 'the work in question', 'said film',
                           'this particular piece', 'the text under consideration'],
        r'\bI\b': ['this viewer', 'the present reviewer', 'one', 'the undersigned',
                   'this particular audience member'],
        r'\bmy\b': ['this viewer\'s', 'one\'s', 'the present reviewer\'s',
                    'the subjective position of this audience member regarding'],
    }
    
    for pattern, options in obfuscations.items():
        if re.search(pattern, sentence, re.IGNORECASE):
            if random.random() < 0.7:
                replacement = random.choice(options)
                sentence = re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
    
    return sentence


def discourse_marker_overload(sentence):
    """Add 3-4 discourse markers to create maximum verbosity."""
    
    opening_markers = [
        "Frankly speaking, ",
        "To be entirely candid, ",
        "In the interest of full disclosure, ",
        "Speaking with complete honesty, ",
        "From a position of analytical objectivity, ",
        "Without reservation or equivocation, ",
    ]
    
    middle_qualifiers = [
        ", it must be emphasized that, ",
        ", and it should be understood that, ",
        ", while simultaneously acknowledging that, ",
        ", though it merits notation that, ",
        ", and one must recognize that, ",
    ]
    
    epistemic_markers = [
        "It is the case that ",
        "The factual situation is that ",
        "What can be stated with confidence is that ",
        "The reality of the matter is that ",
        "It must be acknowledged that ",
    ]
    
    concluding_markers = [
        ", ultimately speaking",
        ", when all factors are considered",
        ", in the final analysis",
        ", taking the totality into account",
        ", upon comprehensive reflection",
    ]
    
    # Add opening
    if random.random() < 0.7:
        sentence = random.choice(opening_markers) + sentence[0].lower() + sentence[1:]
    
    # Add epistemic marker (create complex embedding)
    if random.random() < 0.6:
        parts = sentence.split(',', 1)
        if len(parts) == 2:
            sentence = parts[0] + ', ' + random.choice(epistemic_markers).lower() + parts[1]
    
    # Add middle qualifier
    if random.random() < 0.5 and ',' in sentence:
        parts = sentence.rsplit(',', 1)
        if len(parts) == 2:
            sentence = parts[0] + random.choice(middle_qualifiers) + parts[1].lstrip()
    
    # Add concluding marker
    if random.random() < 0.6:
        sentence = sentence.rstrip('.!') + random.choice(concluding_markers) + '.'
    
    return sentence


def syntactic_scrambling(sentence):
    """Reorder clauses and use marked word orders."""
    
    # Fronting: "X was Y" → "Y was X" type transformations
    if ' was ' in sentence and random.random() < 0.5:
        parts = sentence.split(' was ', 1)
        if len(parts) == 2 and len(parts[0].split()) < 5:
            # Create cleft: "X was Y" → "What was Y was X"
            sentence = f"What was {parts[1].rstrip('.!')} was {parts[0].lower()}."
    
    # Inversion patterns for emphasis
    if re.search(r'\b(good|great|bad|terrible) (was|were)', sentence, re.IGNORECASE):
        match = re.search(r'\b(good|great|bad|terrible) (was|were)', sentence, re.IGNORECASE)
        adj = match.group(1)
        verb = match.group(2)
        # "The acting was great" → "Great, indeed, was the acting"
        sentence = re.sub(
            r'(\w+) ' + verb + r' ' + adj,
            f"{adj.capitalize()}, indeed, {verb} the \\1",
            sentence,
            flags=re.IGNORECASE
        )
    
    return sentence


def fragment_sentiment(sentences):
    """Break single sentiment expressions across multiple sentences."""
    
    if len(sentences) < 2:
        return sentences
    
    result = []
    skip_next = False
    
    for i, sent in enumerate(sentences):
        if skip_next:
            skip_next = False
            continue
        
        # Look for sentiment + conjunction patterns to split
        if random.random() < 0.6 and (' but ' in sent or ' and ' in sent or ' however ' in sent.lower()):
            # Split compound sentiment sentences
            if ' but ' in sent:
                parts = sent.split(' but ', 1)
                result.append(parts[0].strip().rstrip(',') + '.')
                result.append('However, upon further consideration, ' + parts[1].strip())
            elif ' and ' in sent and ('good' in sent.lower() or 'bad' in sent.lower() or 'great' in sent.lower()):
                parts = sent.split(' and ', 1)
                result.append(parts[0].strip().rstrip(',') + '.')
                result.append('Additionally, it must be noted that ' + parts[1].strip())
            else:
                result.append(sent)
        else:
            result.append(sent)
    
    return result
        

def augment_text_with_transformation(texts, labels, transform_fn, num_augmented):
    """
    Create augmented dataset by applying transformation
    
    Args:
        texts: List of original texts
        labels: List of original labels
        transform_fn: Transformation function to apply
        num_augmented: Number of augmented samples to create
    
    Returns:
        augmented_texts, augmented_labels
    """
    indices = random.sample(range(len(texts)), min(num_augmented, len(texts)))
    
    augmented_texts = []
    augmented_labels = []
    
    print(f"Creating {len(indices)} augmented examples...")
    for idx in indices:
        transformed_text = transform_fn(texts[idx])
        augmented_texts.append(transformed_text)
        augmented_labels.append(labels[idx])
    
    return augmented_texts, augmented_labels