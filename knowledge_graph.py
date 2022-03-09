# -*- coding: utf-8 -*-
import json
import rdflib
from rdflib import Literal
from rdflib.namespace import CSVW, DC, DCAT, DCTERMS, DOAP, FOAF, ODRL2, ORG, OWL, PROF, PROV, RDF, RDFS, SDO, SH, SKOS, \
    SOSA, SSN, TIME


def func():
    with open("PsyQA_full.json", "rb") as jsonfile:
        data = json.load(jsonfile)
    g = rdflib.Graph()

    i = 0
    for info in data:

        # 实体
        subject = rdflib.URIRef('http://www.example.org/' + str(i))
        # 关系
        question = rdflib.URIRef('http://www.example.org/question')
        description = rdflib.URIRef('http://www.example.org/description')
        embedding = rdflib.URIRef('http://www.example.org/embedding')
        answer = rdflib.URIRef('http://www.example.org/answer')
        # 添加三元组
        g.add((subject, question, Literal(info['question'])))
        g.add((subject, description, Literal(info['description'])))
        g.add((subject, embedding, Literal(0.0)))
        for answer_text in info['answers']:
            g.add((subject, answer, Literal(answer_text['answer_text'])))
        g.serialize('kg.rdf', format='turtle')
        i += 1
    # print(g.serialize(format='json-ld'))
    kg = open("kg.rdf", 'w', encoding='utf-8')
    kg.write(g.serialize(format='json-ld'))


if __name__ == '__main__':
    func()
