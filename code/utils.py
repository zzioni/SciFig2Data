from bs4 import BeautifulSoup
import spacy
import json
import os


def parse_xml(input_file):

    #python -m spacy download en_core_web_lg

    fig_json = {}

    nlp = spacy.load("en_core_web_lg")

    xml_data = open(input_file, "r", encoding="utf-8").read()
    soup = BeautifulSoup(xml_data, 'xml')

    all_without_section = soup.select("body > p")
    if all_without_section != []:
        for p in all_without_section:

            all_fig = p.find_all("fig")
            for fig in all_fig:
                try:
                    fig_label = fig.find('label').text
                except:
                    continue

                fig_id = fig['id']
                fig_caption = fig.find('caption').text
                fig_name = fig.find('graphic')['xlink:href']

                fig_sent = []
                if p.find('xref', rid=fig_id) != None:
                    fig_text_in_sent = p.find('xref', rid=fig_id).text
                    p_text = p.text
                    sent_list = nlp(p_text)
                    for sent in sent_list.sents:
                        sent = sent.text
                        if fig_text_in_sent.isdigit():
                            if fig_label in sent:
                                fig_sent.append(sent)
                        else:
                            if fig_text_in_sent in sent:
                                fig_sent.append(sent)
                pre_dic = {'caption': fig_caption, 'section':'' , 'intext':fig_sent}
                fig_json[fig_name]  = pre_dic

    all_body_section = soup.select("body > sec")

    for root_section in all_body_section:

        root_section_title = root_section.find("title").text
        child_section_list = root_section.select('sec > sec')

        if child_section_list == []:
            all_fig = root_section.find_all("fig")
            for fig in all_fig:
                try:
                    fig_label = fig.find('label').text
                except:
                    continue

                fig_id = fig['id']
                fig_caption = fig.find('caption').text
                fig_name = fig.find('graphic')['xlink:href']
                child_section_p = root_section.find_all('p')
                fig_sent = []
                for p in child_section_p:
                    if p.find('xref', rid=fig_id) != None:
                        fig_text_in_sent = p.find('xref', rid=fig_id).text
                        p_text = p.text
                        sent_list = nlp(p_text)
                        for sent in sent_list.sents:
                            sent = sent.text
                            if fig_text_in_sent.isdigit():
                                if fig_label in sent:
                                    fig_sent.append(sent)
                            else:
                                if fig_text_in_sent in sent:
                                    fig_sent.append(sent)

                pre_dic = {'caption': fig_caption, 'section':root_section_title , 'intext':fig_sent}
                fig_json[fig_name] = pre_dic

        else:
            for child_section in child_section_list:
                child_section_title = child_section.find('title').text

                all_fig = child_section.find_all("fig")
                for fig in all_fig:
                    try:
                        fig_label = fig.find('label').text
                    except:
                        continue

                    fig_id = fig['id']
                    fig_caption = fig.find('caption').text
                    fig_name = fig.find('graphic')['xlink:href']

                    child_section_p = child_section.find_all('p')
                    fig_sent = []
                    for p in child_section_p:
                        if p.find('xref', rid=fig_id) != None:
                            fig_text_in_sent = p.find('xref', rid=fig_id).text
                            p_text = p.text
                            sent_list = nlp(p_text)
                            for sent in sent_list.sents:
                                sent = sent.text
                                if fig_text_in_sent.isdigit():
                                    if fig_label in sent:
                                        fig_sent.append(sent)
                                else:
                                    if fig_text_in_sent in sent:
                                        fig_sent.append(sent)
                    pre_dic = {'caption': fig_caption, 'section':root_section_title + '|' + child_section_title, 'intext':fig_sent}
                    fig_json[fig_name] = pre_dic
 
    fig_json = json.dumps(fig_json)  # Convert json to dict

    return fig_json


