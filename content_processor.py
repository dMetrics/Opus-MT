import re

from ucenum import ucenum
from ucinfo import ucinfo

from apply_bpe import BPE
from mosestokenizer import MosesSentenceSplitter, MosesPunctuationNormalizer, MosesTokenizer, MosesDetokenizer
import sentencepiece

UNICODE_TERMINATORS = "".join(
    c.printable for c in map(ucinfo, ucenum('P'))
    if c.printable != '.' and c.name.endswith("FULL STOP")
    or "INVERTED" not in c.name and
    ("QUESTION MARK" in c.name or "EXCLAMATION MARK" in c.name)
)


class SentSplitHelper:

    @staticmethod
    def split(sentences):
        result = []
        for s in sentences:
            parts = re.split(f'([{UNICODE_TERMINATORS}])', s)
            if len(parts) == 1:
                result.append(s)
            else:
                for new_s, term in zip(parts[:-1:2], parts[1::2]):
                    result.append(new_s + term)
                if len(parts) % 2 > 0:
                    result.append(parts[-1])
        return result


class ContentProcessor():
    def __init__(self,  srclang, targetlang, sourcebpe=None, targetbpe=None,sourcespm=None,targetspm=None):
        self.bpe_source = None
        self.bpe_target = None
        self.sp_processor_source = None
        self.sp_processor_target = None
        self.sentences=[]
        # load BPE model for pre-processing
        if sourcebpe:
            # print("load BPE codes from " + sourcebpe, flush=True)
            BPEcodes = open(sourcebpe, 'r', encoding="utf-8")
            self.bpe_source = BPE(BPEcodes)
        if targetbpe:
            # print("load BPE codes from " + targetbpe, flush=True)
            BPEcodes = open(targetbpe, 'r', encoding="utf-8")
            self.bpe_target = BPE(BPEcodes)

        # load SentencePiece model for pre-processing
        if sourcespm:
            # print("load sentence piece model from " + sourcespm, flush=True)
            self.sp_processor_source = sentencepiece.SentencePieceProcessor()
            self.sp_processor_source.Load(sourcespm)
        if targetspm:
            # print("load sentence piece model from " + targetspm, flush=True)
            self.sp_processor_target = sentencepiece.SentencePieceProcessor()
            self.sp_processor_target.Load(targetspm)

        # pre- and post-processing tools
        self.tokenizer = None
        self.detokenizer = None

        # TODO: should we have support for other sentence splitters?
        # print("start pre- and post-processing tools")
        # additional options more and even_more to split chinese texts
        self.sentence_splitter = MosesSentenceSplitter(srclang, more=True)
        self.sentence_splitter_helper = SentSplitHelper()
        self.normalizer = MosesPunctuationNormalizer(srclang)
        if self.bpe_source:
            self.tokenizer = MosesTokenizer(srclang)
            self.detokenizer = MosesDetokenizer(targetlang)

    def preprocess(self, srctxt):
        lines = []
        split_lines = srctxt.split('\n')
        for line in split_lines:
            normalized = self.normalizer(line)
            if not (type(normalized) == list):
                normalized = [normalized]
            split = self.sentence_splitter(normalized)
            split = self.sentence_splitter_helper.split(split)
            if split and line != split_lines[-1]:
                split[len(split) - 1] = split[len(split) - 1] + "\n"
            lines.extend(split)
        # normalized_text = '\n'.join(lines)   # normalizer do not accept '\n'
        nl_positions = []
        for s in lines:
            if "\n" in s:
                nl_positions.append(lines.index(s))
        sentSource = lines

        self.sentences = []
        for s in sentSource:
            if self.tokenizer:
                # print('raw sentence: ' + s, flush=True)
                tokenized = ' '.join(self.tokenizer(s))
                # print('tokenized sentence: ' + tokenized, flush=True)
                segmented = self.bpe_source.process_line(tokenized)
            elif self.sp_processor_source:
                print('raw sentence: ' + s, flush=True)
                segmented = ' '.join(self.sp_processor_source.EncodeAsPieces(s))
                # print(segmented, flush=True)
            else:
                raise RuntimeError("No tokenization / segmentation method defines, can't preprocess")
            self.sentences.append(segmented)
        return self.sentences, nl_positions

    def postprocess(self, recievedsentences, nl_indices=None):
        sentTranslated = []
        for index, s in enumerate(recievedsentences):
            received = s.strip().split(' ||| ')
            # print(received, flush=True)

            # undo segmentation
            if self.bpe_source:
                translated = received[0].replace('@@ ','')
            elif self.sp_processor_target:
                translated = self.sp_processor_target.DecodePieces(received[0].split(' '))
            else:
                translated = received[0].replace(' ','').replace('▁',' ').strip()

            alignment = ''
            if len(received) == 2:
                alignment = received[1]
                links = alignment.split(' ')
                fixedLinks = []
                outputLength = len(received[0].split(' '))
                for link in links:
                    ids = link.split('-')
                    if ids[0] != '-1' and int(ids[0])<len(self.sentences[index]):
                        if int(ids[1])<outputLength:
                            fixedLinks.append('-'.join(ids))
                alignment = ' '.join(fixedLinks)

            if self.detokenizer:
                detokenized = self.detokenizer(translated.split())
            else:
                detokenized = translated

            sentTranslated.append(detokenized)
        if nl_indices:
            for pos in nl_indices:
                sentTranslated[pos] = sentTranslated[pos] + "\n"
        return sentTranslated
