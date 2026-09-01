_skript_data = [
    {'query': 'Was ist ein Product Owner?',
     'ground_truth': 'Er ist er Kunde und legt das gemeinsame Ziel fest, das das Team zusammen mit ihm erreichen muss, und in den jeweiligen Iterationsschritten die Prioritäten der entsprechenden Teilprodukt-Anforderungen (Product Backlogs, s.u.). Zur Definition der Ziele dienen ihm (User) Stories , d.h. in Alltagssprache mit maximal zwei Sätzen bewusst kurz gefasste Software-Anforderungen, oder andere Techniken wie UMLAnwendungsfälle.'},

    {'query': 'Was macht der Scrum Master?',
     'ground_truth': 'Er überwacht die Aufteilung der Rollen und Rechte zu überwachen, hält die Transparenz während der gesamten Entwicklung aufrecht und unterstützt dabei, Verbesserungspotentiale zu erkennen und zu nutzen. Er steht dem Team zur Seite, 4. ist aber weder Product Owner noch Teil des Teams. Er versucht, für die ordnungsgemäße Durchführung und Implementierung im Rahmen des Projektes zu sorgen. Er hat die Pflicht, darauf zu achten, dass der Product Owner nicht in den adaptiven Selbstorganisationsprozess des Teams eingreift.'},

    {'query': 'Von wem wurde das Spiralmodell eingeführt?',
     'ground_truth': 'Das Spiralmodell wurde von Boehm eingeführt.'},

    {'query': 'Was ist Paarprogrammierung?',
     'ground_truth': "Ein ganz neuartiges Prinzip der agilen Softwareentwicklung ist die Paarprogrammierung (pair programming) . Hierbei wird der Quelltext von jeweils zwei Programmierern an einem Rechner erstellt."},

    {'query': "Welche Aufgaben haben Driver und Navigator bei der Paarprogrammierung?",
     'ground_truth': "Ein Programmierer, der „Driver“, schreibt dabei den Code, während der andere, der „Navigator“, über die Problemstellungen nachdenkt, den geschriebenen Code kontrolliert und Probleme, die ihm dabei auffallen, sofort anspricht, z.B. Schreibfehler oder logische Fehler."},

    {'query': "Was sind die Grundprinzipien von eXtreme Programmierung (XP)?",
     'ground_truth': "Oberstes Prinzip der eXtremen Programmierung ist Einfachheit Neben der Einfachheit sind effektive Kommunikation und sofortiges Feedback Grundprinzipien des XP."},

    {'query': "Welche Probleme oder Konflikte können beim Mergen auftreten?",
     'ground_truth': "Werden jedoch verschiedene Änderungen zusammengeführt, die den gleichen Abschnitt einer Datei betreffen, so kommt es zu einem Merge-Konflikt."},

    {'query': "Wie kann ein Merge-Konflikt gelöst werden?",
     'ground_truth': "Dieser kann nur manuell aufgelöst werden."},

    {'query': "Welche Möglichkeiten gibt es eine Version einer Resource zu erstellen?",
     'ground_truth': "einerseits im lokalen Workspace und danach mit commit im Repsoitory, oder direkt in einem Branch im Repository."},

    {'query': "Was war eine der grundlegenden Annahmen der Unix-Entwicklung",
     'ground_truth': "dass der Benutzer mit dem Computer umgehen kann, dass er weiß was er tut."},

    {'query': "Was macht ein Vorgehensmodell?",
     'ground_truth': "Ein Vorgehensmodell , auch: Lebenszyklusmodell , gliedert im Allgemeinen einen Gestaltungs- oder Produktionsprozess in strukturierte Aktivitäten oder Phasen , denen wiederum entsprechende Methoden und Techniken der Organisation zugeordnet sind. Aufgabe eines Vorgehensmodells ist es, die allgemein in einem Gestaltungsprozess auftretenden Aufgabenstellungen und Aktivitäten in ihrer logischen Ordnung darzustellen"},

    {'query': "Welche Analysemetriken gibt es?",
     'ground_truth': "Die Metrik DRY (don’t repeat yourself) der Metrik STY (styleness)"},

    {'query': "Welche Regel gilt beim Testen von objektorientierten Sprachen?",
     'ground_truth': "'Stets das Benutzte vor dem Benutzenden testen.' D.h. bei Assoziationen und Kompositionen wird die verwendete vor der verwendenden Klasse getestet, bei Vererbungshierarchien die Superklasse vor der erbenden Klasse"},
]

def get_skript_data() -> list[dict[str, str]]:
    for i, item in enumerate(_skript_data):
        item["id"] = "sk" + str(i)
    return _skript_data


_skript_formula_data = []

def get_skript_formula_data() -> list[dict[str, str]]:
    for i, item in enumerate(_skript_formula_data):
        item["id"] = "sf" + str(i)
    return _skript_formula_data

_slides_data = []

def get_slides_data() -> list[dict[str, str]]:
    for i, item in enumerate(_slides_data):
        item["id"] = "sl" + str(i)
    return _slides_data