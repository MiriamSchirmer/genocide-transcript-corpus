#################################################################################
### IMPORTS
import re

#################################################################################
### GLOBALS
GLB_EMPTY_STRING = ""
RE_GLB_CASE = re.IGNORECASE
RE_PARAGRAPH_WITH_B_TAG_AND_WHITE_SPACES = r'<P><B>( )*</B></P>'
RE_PARAGRAPH_WITH_B_TAG_AND_NUM_PAGES = r'<P><B>( )*Page (\d)+( )*</B></P>'
RE_PARAGRAPH_WITH_TEXT_BLANK_PAGE_INSERTED = r'(<P>( \d+|\d+)( )*Blank page inserted to ensure pagination corresponds between the French and( )*</P>)|(<P>( \d+|\d+)( )*English transcripts.( )*</P>)'
RE_PARAGRAPH_WITH_NUMBERS_AND_SPACES = r'<P>( \d+|\d+)( )*</P>'
RE_PARAGRAPH_WITH_SPACES_AT_THE_END = r'</P>( )*'
RE_SPANS_AT_THE_END = r'</span>( )*'
RE_PARAGRAPH_WITH_CLASS_AND_STYLE = r'<P class=\"(.)*\" style=\"(.)*\">( |( )*)\d+( )*|<p>'
RE_SPANSTYLE = r'<spanstyle=(\"|\')(.)*(\"|\')>'
RE_SPANSTYLE_CLOSE = r'</spanstyle=(.)*>'
RE_SPAN_WITH_CLASS_AND_STYLE = r'<span( )+style=(\"|\')(.)*(\"|\')>'
RE_B_BEGIN_OR_END = r'<B>|</B>'
RE_PARAGRAPH_EXCEPTION_CLASS_STYLE = r'<P class=\"(.)*\" style=\"(.)*\">'
RE_SPAN_LANG_FR = r'<span lang=\"FR\" style=(.)*>\d+( )*Page(s?) (\d+|\d+-\d+) redacted\. (Private|Closed) session\.'
RE_PARAGRAPH_WITH_NUMBERS_AND_SPACES_AT_THE_BEGINNING = r'<P>( )*\d+( )*'
RE_SPECIFIC_TAGS = r'<HTML>|</HTML>|<center>|</center>|</FONT>|</BODY>|<FONT FACE(.*)>'
RE_PARAGRAPH_WITH_PAGE_CLOSED_SESSION = r'page (\d+) redacted – closed session'
RE_PARAGRAPH_WITH_REDACTED = r'\(redacted\)'
RE_PARAGRAPH_WITH_OPEN_OR_CLOSED_SESSION = r'\((Open|Close) session\)'
RE_PARAGRAPH_WITH_SHORT_ADJOURMENT = r'\(Short Adjournment\)'
RE_PARAGRAPH_WITH_TIMESTAMP = r'\(\d+\.\d+ (p\.m\.|a\.m\.)\)(.|( )*)|\(\d+\.\d+\)'
RE_SENTENCE_BEGIN_WITH_NUMBER_AND_SPACES = r'( )*\d{1,2}( )*(?)+'
RE_SENTENCE_PAGE_NUMBER = r'Page \d+'
GLB_ECCC_ROW_RANGE_BEGIN = 1
GLB_ECCC_ROW_RANGE_END = 25
GLB_ECCC_PATTERN_BEGIN_CONTENT_OF_INTEREST_OPT1 = "P R O C E E D I N G S"
GLB_ECCC_PATTERN_BEGIN_CONTENT_OF_INTEREST_OPT2 = "PROCEEDINGS"
GLB_ECCC_PATTERN_BEGIN_CONTENT_LIST = [GLB_ECCC_PATTERN_BEGIN_CONTENT_OF_INTEREST_OPT1, GLB_ECCC_PATTERN_BEGIN_CONTENT_OF_INTEREST_OPT2]
RE_ECCC_SENT_NUMBER_AT_THE_BEGINNING = r'(?m)^(\d+)'#r'( )*\d+( )+'#r'^(\w+|^( ))( )*\d+( )+'
RE_ECCC_SENT_IDS_HEADER = r'\w+\d+\/\d+\.\d+'
RE_ECCC_SENT_TIMESTAMPS = r'\[\d{1,2}(\.|\:)\d{2}(\.|\:)\d{2}\]'
RE_ICTR_SENT_DATE = r'\d{2}( )\w{3}( )\d{2}'
RE_ICTR_SENT_JUST_NUMBERS = r'( )*\d+( )+\n'

#################################################################################
### Cleaning of transcripts of the "International Criminal Tribunal of the 
### Former Yugoslavia"
#== @input string with the content in html format. Specifically, paragraphs 
#           between the tags "<p>" and "</p>" are received
#== @return 
def cleanParagraphsICFYtranscript(src_content):
    # Remove content that is within "b" tag with num of pages
    final_content = re.sub(RE_PARAGRAPH_WITH_B_TAG_AND_NUM_PAGES, GLB_EMPTY_STRING, src_content, flags=RE_GLB_CASE)
    #
    final_content = re.sub(RE_SPANSTYLE, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)    
    #
    final_content = re.sub(RE_SPAN_WITH_CLASS_AND_STYLE, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)    
    # Remove spaces at the end of the paragraph
    final_content = re.sub(RE_PARAGRAPH_WITH_SPACES_AT_THE_END, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    #
    final_content = re.sub(RE_SPANS_AT_THE_END, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)    
    #
    final_content = re.sub(RE_B_BEGIN_OR_END, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove numbers and and spaces at the beginning of the paragraph
    final_content = re.sub(RE_PARAGRAPH_WITH_NUMBERS_AND_SPACES_AT_THE_BEGINNING, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    #
    final_content = re.sub(RE_PARAGRAPH_WITH_CLASS_AND_STYLE, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove "page \d+ redacted - closed session"
    final_content = re.sub(RE_PARAGRAPH_WITH_PAGE_CLOSED_SESSION, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove "(redacted)"
    final_content = re.sub(RE_PARAGRAPH_WITH_REDACTED, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove open or closed session
    final_content = re.sub(RE_PARAGRAPH_WITH_OPEN_OR_CLOSED_SESSION, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove short adjourment
    final_content = re.sub(RE_PARAGRAPH_WITH_SHORT_ADJOURMENT, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove timestaps within the transcript
    final_content = re.sub(RE_PARAGRAPH_WITH_TIMESTAMP, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    #
    final_content = re.sub(RE_PARAGRAPH_EXCEPTION_CLASS_STYLE, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    #
    final_content = re.sub(RE_SPAN_LANG_FR, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    #
    final_content = re.sub(RE_SPANSTYLE_CLOSE, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove spaces before and after the current content
    final_content = final_content.strip()
    # Possible:
    # - (Luncheon Adjournment)

    return final_content

#################################################################################
### Cleaning of transcripts of the "Extraordinary Chamber in the Courts 
### of Cambodia"
#== @input string with each row
#== @return string after cleaning
def cleanSentenceECCCtranscript(src_content):
    final_content = ""

    src_content = src_content.strip()
    # Remove numbers that are located at the beginning of the sentence
    final_content = re.sub(RE_ECCC_SENT_NUMBER_AT_THE_BEGINNING, GLB_EMPTY_STRING, src_content, flags=RE_GLB_CASE)
    # Remove IDs on the header
    final_content = re.sub(RE_ECCC_SENT_IDS_HEADER, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove timestamps
    final_content = re.sub(RE_ECCC_SENT_TIMESTAMPS, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)

    return final_content.strip()

#################################################################################
### Cleaning PDF of the "Extraordinary Chamber in the Courts of Cambodia"
#== @input string with the content extracted from a PDF file.
#== @return string after cleaning
"""
Evidences for implementation:
    - The analysed PDF file contains information from the 4th page (index 3 in PyPDF implementation)
    - All pages contain a header (except first page). Thus, we can ommit the first 10 rows (at least)
    - All pages with the information of our interest contain enumerated rows from 1 to 25
    - First page with information of our interest contain the text "P R O C E E D I N G S". We can iterate over pages and start from the one this pattern is found
    - Pages with information of our interest contain the number on the top.
"""
def cleanPagePdfECCCtranscript(src_content, page_content):
    list_content = list()

    try:
        # Remove timestamps
        result = re.sub(RE_ECCC_SENT_TIMESTAMPS, GLB_EMPTY_STRING, src_content, flags=RE_GLB_CASE)
        result = result[result.index('Page'):]
        result = re.sub(RE_SENTENCE_PAGE_NUMBER, GLB_EMPTY_STRING, result, flags = RE_GLB_CASE)
        result = result[result.index(str(page_content))+len(str(page_content)):]

        for index in range(GLB_ECCC_ROW_RANGE_BEGIN, GLB_ECCC_ROW_RANGE_END+1):
            if index != GLB_ECCC_ROW_RANGE_END:
                begin_index = 0
                end_index = 0

                list_ocurrences_begin = [ocurrence for ocurrence in re.finditer(r'( ){1}' + re.escape(str(index)) + r'( ){1}', result)]
                begin_index = list_ocurrences_begin[0].start()#result.index(str(index))
                list_ocurrences_end = [ocurrence for ocurrence in re.finditer(r'( ){1}' + re.escape(str(index+1)) + r'( ){1}', result)]
                end_index = list_ocurrences_end[0].start() #result.index(str(index+1)) #Req. improvement or validation

                sent = result[begin_index: end_index]
                list_content.append(sent)
                result = result[result.index(sent)+len(sent):]
            else:
                list_content.append(result)
    except Exception as e:
        list_content = None

    return list_content


#################################################################################
### Cleaning PDF of the "International Criminal Tribunal for Rwanda"
#== @input string with the content extracted from a PDF file.
#== @return string after cleaning
"""
Evidences for implementation:
    - ICTR -CHAMBER is repeated in the case was shared. Still pending of validation whether this happens with all cases. Last page of the analysed document presents a pattern with a difference of one character
    - There's a word at the beginning of the document that should be removed. However, this may be different from case to case
    - WARNING: Extraction of information shows poor results.
"""
def cleanPagePdfICTRtranscript(src_content):
    # Remove the dates inserted in the transcript
    result = re.sub(RE_ICTR_SENT_DATE, GLB_EMPTY_STRING, src_content, flags=RE_GLB_CASE)
    # Remove rows with single numbers and newline
    result = re.sub(RE_ICTR_SENT_JUST_NUMBERS, GLB_EMPTY_STRING, result, flags=RE_GLB_CASE)
    # Remove numbers at the beginnig of the sentence
    result = re.sub(RE_ECCC_SENT_NUMBER_AT_THE_BEGINNING, GLB_EMPTY_STRING, result, flags=RE_GLB_CASE)

    return result.strip()