from jnius import autoclass

# Load SimpleNLG classes
NLGFactory = autoclass('simplenlg.framework.NLGFactory')
Realiser = autoclass('simplenlg.realiser.english.Realiser')
SPhraseSpec = autoclass('simplenlg.phrasespec.SPhraseSpec')
Lexicon = autoclass('simplenlg.lexicon.Lexicon')
EnglishLexicon = autoclass('simplenlg.lexicon.EnglishLexicon')

# Create a lexicon, factory, and realiser
lexicon = Lexicon.getDefaultLexicon()
nlgFactory = NLGFactory(lexicon)
realiser = Realiser(lexicon)

# Create a simple sentence
sentence = nlgFactory.createClause()
sentence.setSubject("the cat")
sentence.setVerb("chase")
sentence.setObject("the mouse")

# Realise the sentence
output = realiser.realiseSentence(sentence)

# Print the output
print(output)
