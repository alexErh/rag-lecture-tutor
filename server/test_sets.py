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


_skript_formula_data = [
    {"query": "Was sind Lichtwellenleiter?",
     "ground_truth": r"Lichtwellenleiter (LWL) sind hochreine Quarzglasfasern, die optische Signale weiterleiten. Sie sind eine Weiterentwicklung der Lichtleiter , die das Licht durch vielfache Totalreflexion ¨ ubertragen."},

    {"query":"Wie lautet die Formel / Gleichung für den Übergang von Licht von einem dünneren Medium in ein dickeres? ",
     "ground_truth": r"$$\frac { \sin \alpha } { \sin \beta } = c o n s t . = \frac { n _ { 2 } } { n _ { 1 } } = \frac { c _ { 1 } } { c _ { 2 } }$$"},

    {"query": "Wie berechnet sich die Gleichspannung die bei dem Halleffekt entsteht?",
     "ground_truth": r"$$U _ { H } \sim \frac { I \cdot B } { d }$$"},

    {"query": "Welche Bedingungen unterscheiden lineare (ohmsche) von nichtlinearen Widerständen?",
     "ground_truth": r"Ist der Proportionalit¨ atsfaktor G in (1.37) eine Konstante, dann nimmt der Strom I mit der Spannung linear zu. Ist der Proportionalit¨ atsfaktor nicht konstant, sondern von der Spannung oder vom Strom abh¨ angig, dann spricht man von nichtlinearen Widerst¨ anden."},

    {"query": "Wie wird der Leitwert G definiert und in welcher Einheit wird er angegeben?",
     "ground_truth": r"$$\begin{matrix} I \sim U \\ \text {order} & I = G \cdot U \end{matrix} \quad ( 1 . 3 7 )$$ Der Proportionalit¨ atsfaktor G wird Leitwert genannt. Die Einheit ist Siemens (S)"},

    {"query": "Was ist elektrische Leistung und wie ist sie definiert?",
     "ground_truth": r"Leistung ist definiert als Arbeit pro Zeiteinheit. F¨ ur die elektrische Leistung folgt deshalb: $$P = \frac { W } { t } = U \cdot I = I ^ { 2 } \cdot R = \frac { U ^ { 2 } } { R }$$"},

    {"query": "Wie berechnet man die Teilströme in einer Parallelschaltung von Widerständen?",
     "ground_truth": r"Der Gesamtstrom I verzweigt sich in die Teilstr¨ ome I 1 , I 2 , · · · , I n , die durch die zugeh¨ origen Widerst¨ ande R 1 , R 2 , · · · , R n fließen. Die Teilstr¨ ome werden nach dem Ohmschen Gesetz berechnet: $$I _ { 1 } = \frac { U } { R _ { 1 } } \ I _ { 2 } = \frac { U } { R _ { 2 } } \ \dots \ I _ { n } = \frac { U } { R _ { n } }$$"},

    {"query": "Welche drei physikalischen Wirkungen werden bei magneto-optischen Speichermedien für Schreiben, Speichern und Lesen genutzt?",
     "ground_truth": r"Drei Reaktionsf¨ ahigkeiten werden bei magneto-optischen Speichermedien genutzt: 1. eine thermomagnetische f¨ ur den Schreibvorgang 2. eine magnetische f¨ ur die permanente Speicherung 3. eine magneto-optische f¨ ur den Lesevorgang"},

    {"query": "Was ist ein CMOS-Transmissiongate und woraus besteht es?",
     "ground_truth": r"Bei diesem Schaltkreis zur bidirektionalen Signal¨ ubertragung sind ein NMOS-FET und ein PMOS-FET parallel geschaltet"},

    {"query": "Wie codiert das Quine-McCluskey-Verfahren Minterme in der beschriebenen Schreibweise? Und gebe mir ein Beispiel",
     "ground_truth": r"Die Minterme der Schaltfunktion werden nicht durch die negierten und nichtnegierten Variablen dargestellt, sondern durch ihr Bin¨ ar¨ aquivalent : 1steht f¨ ur eine nicht negierte Variable 0steht f¨ ur eine negierte Variable -steht f¨ ur eine nicht auftretende Variable $$\begin{array} { r l r } { \text {Beispie} \colon } & { x _ { 4 } \wedge \overline { x } _ { 3 } \wedge x _ { 2 } \wedge \overline { x } _ { 1 } } & { \hat { = } } & { 1 \, 0 \, 1 \, 0 } \\ & { x _ { 4 } \wedge \overline { x } _ { 3 } \wedge \overline { x } _ { 1 } } & { \hat { = } } & { 1 \, 0 \, - \, 0 } \\ & { x _ { 4 } \wedge x _ { 2 } } & { \hat { \equiv } } & { 1 \, - \, 1 \, - } \end{array}$$"},

    {"query": "Welche Werte haben Strom I und Spannung UQ im Schalterzustand „ein“ bzw. „aus“?",
     "ground_truth": r"- -Im Schalterzustand ein ist der Innenwiderstandswert des Schalters S R i = 0. Daraus folgt I = U B /R und U Q =0V. - -Im Schalterzustand aus ist der Sperrwiderstand des Schalters S R s = ∞ . Daraus folgt I = 0 und U Q = U B ."},

    {"query": "Was ist eine Schaltfunktion?",
     "ground_truth": r"wird durch die Zuordnungsvorschrift f eindeutig ein Funktionswert $$f ( x _ { 1 } , x _ { 2 } , \dots , x _ { n } ) \in \{ 0 , 1 \} \, z u g e o r d n e t .$$ Man schreibt $$y = f ( x _ { 1 } , x _ { 2 } , \dots , x _ { n } )$$ und sagt y ist eine Funktion von x 1 , x 2 , . . . , x n . Der Ausdruck y = f ( x 1 , x 2 , . . . , x n ) wird Schaltfunktion genannt"},

]

def get_skript_formula_data() -> list[dict[str, str]]:
    for i, item in enumerate(_skript_formula_data):
        item["id"] = "sf" + str(i)
    return _skript_formula_data

_slides_data = [
    {
        "query": "Wie definiert die DIN 69901 den Begriff \"Projekt\"?",
        "ground_truth": "Ein Projekt ist ein Vorhaben, das im wesentlichen durch die Einmaligkeit der Bedingungen in ihrer Gesamtheit gekennzeichnet ist, z.B.\nZielvorgabe\nzeitliche, finanzielle, personelle und andere Begrenzungen\nAbgrenzung gegenüber anderen Vorhaben\nprojektspezifische Organisation.\n(DIN 69 901)",
    },  # S. 12
    {
        "query": "Wie definiert die DIN 69901 den Begriff \"Projektmanagement\"?",
        "ground_truth": "\" Gesamtheit von Führungsaufgaben, -organisation, -techniken und - mitteln für die Abwicklung eines Projektes '",
    },  # S. 14
    {
        "query": "Welche vier Zielgrößen bilden das Teufelsquadrat nach Harry Sneed?",
        "ground_truth": "Zeit: Projektlaufzeit\nKosten: Budget\nQualität: z.B. Funktionalität, Nutzbarkeit, Wartbarkeit\nLeistungsumfang: Anzahl ausgelieferter Funktionen",
    },  # S. 21
    {
        "query": "Was besagt die Invarianz der Fläche im Teufelsquadrat nach Sneed?",
        "ground_truth": "Die Fläche (Produktivität) eines Projekts ist invariant\nWenn ein Projekt z. B. in weniger Zeit und zu geringeren Kosten abgeschlossen werden soll, verringern sich auch Leistungsumfang und Qualität.",
    },  # S. 22
    {
        "query": "Was ist das \"Chinesenprinzip\" und unter welchen Bedingungen funktioniert es?",
        "ground_truth": "'Chinesenprinzip'\nIdee: Ein Projekt wird beschleunigt, indem massiv Personen in das Projekt entsandt werden.\nFunktioniert nur bei stark parallelisierbaren, voneinander unabhängigen Tätigkeiten, die keine größere Einarbeitung erfordern.",
    },  # S. 22
    {
        "query": "Aus wie vielen Aufgabenfeldern besteht Projektmanagement laut Project Management Institute, und wie heißen sie?",
        "ground_truth": "Laut Project Management Institute besteht PM aus den folgenden 9 Aufgabenfeldern:\nIntegrierende Aufgaben (integration management)\nUmfangsmanagement (scope management)\nZeitmanagement (time management)\nKostenmanagement (cost management)\nQualitätsmanagement (quality management)\nPersonalmanagement (human resource management)\nKommunikationsmgmt. (communication management)\nRisikomanagement (risk management)\nBeschaffungsmanagement.(procurement management)",
    },  # S. 25
    {
        "query": "Welche drei Arten der Abgrenzung eines Projekts werden unterschieden?",
        "ground_truth": "Zeitliche Abgrenzung\nSachliche Abgrenzung\nSoziale Abgrenzung",
    },  # S. 13
    {
        "query": "Wofür steht WBS und welchem Zweck dient sie im Umfangsmanagement?",
        "ground_truth": "Erarbeiten einer Produktzerlegung (WBS: Work Breakdown Structure)\nIn welche handhabbaren Teile sollte man das Gesamtprodukt (genauer eigentlich: den konkreten Prozess) zerlegen?\nWie fügen sich diese Teile zum Ganzen zusammen?\nBei SW hauptsächlich: Entwurf + Prozessmodell",
    },  # S. 28
    {
        "query": "Was war 1999 die Ursache für den Verlust der Sonde \"Mars Climate Orbiter\"?",
        "ground_truth": "September 1999: Verlust der Sonde \"Mars Climate Orbiter\" wegen falscher Einheitenumrechnung",
    },  # S. 8
    {
        "query": "Wie wird im Projektmanagement mit den Kosten bzw. dem Budget eines Projekts umgegangen?",
        "ground_truth": "Das Projekt wird mit einem festen Budget gestartet.\nDieses Budget wird nachträglich nicht mehr gekürzt. Auch bei Anbietern von Festpreisprojekten wird in der Regel nicht während der Projektlaufzeit das für das Projekt verfügbare Budget gekürzt, um einen höheren Gewinn zu erzielen.\nDas Budget wird nicht ohne Grund erhöht.",
    },  # S. 23
]

def get_slides_data() -> list[dict[str, str]]:
    for i, item in enumerate(_slides_data):
        item["id"] = "sl" + str(i)
    return _slides_data