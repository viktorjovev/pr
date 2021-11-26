import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

class BayesSpamFilter {

    Map<String, Float> conditionalSpam;
    Map<String, Float> conditionalHam;
    int totalVocabulary;
    int totalWordsSpamMessages;
    int totalWordsHamMessages;
    float probabilityOfSpam;
    float probabilityOfHam;
    int alpha;


    public BayesSpamFilter() {
        conditionalHam = new HashMap<>();
        conditionalSpam = new HashMap<>();
        totalWordsSpamMessages = 0;
        totalWordsHamMessages = 0;
        totalVocabulary = 0;
        probabilityOfSpam = 0.5F;
        probabilityOfHam = 0.5F;
        alpha = 1;
    }

    public void fit(ArrayList<String> xTrain, ArrayList<String> yTrain) {
        List<String> xTrainLowerCase;
        xTrainLowerCase = xTrain.stream().
                map((text) -> text.replaceAll("[^a-zA-Z ]", "")).
                map(String::toLowerCase).
                collect(Collectors.toList());

        Set<String> bagOfWords = new HashSet<>();
        for (String s : xTrainLowerCase) {
            String[] words = s.split("\\s+");
            bagOfWords.addAll(Arrays.asList(words));
        }

        List<String> spamMessages = new ArrayList<>();
        List<String> hamMessages = new ArrayList<>();

        for (int i = 0; i < yTrain.size(); i++) {
            if (yTrain.get(i).equals("spam")) {
                spamMessages.add(xTrainLowerCase.get(i));
                totalWordsSpamMessages += xTrainLowerCase.get(i).length();
            } else {
                hamMessages.add(xTrainLowerCase.get(i));
                totalWordsHamMessages += xTrainLowerCase.get(i).length();
            }
        }
        Map<String, Integer> countsConditionalSpam = new HashMap<>();
        Map<String, Integer> countsConditionalHam = new HashMap<>();
        for (String word : bagOfWords) {

            countsConditionalSpam.putIfAbsent(word, 0);
            for (String text : spamMessages) {
                long counts = Arrays.stream(text.split("\\s+")).filter(w -> w.equals(word)).count();
                Integer temp = countsConditionalSpam.get(word);
                countsConditionalSpam.put(word, (int) (temp + counts));
            }

            countsConditionalHam.putIfAbsent(word, 0);
            for (String text : hamMessages) {
                long counts = Arrays.stream(text.split("\\s+")).filter(w -> w.equals(word)).count();
                Integer temp = countsConditionalHam.get(word);
                countsConditionalHam.put(word, (int) (temp + counts));
            }
        }

        probabilityOfSpam = (float) spamMessages.size() / (float) xTrainLowerCase.size();
        probabilityOfHam = (float) hamMessages.size() / (float) xTrainLowerCase.size();
        totalVocabulary = bagOfWords.size();
        conditionalSpam = bagOfWords.stream().collect(Collectors.toMap(Function.identity(), v -> 0.0F));
        conditionalHam = bagOfWords.stream().collect(Collectors.toMap(Function.identity(), v -> 0.0F));

        for (String word : bagOfWords) {
            int nWordGivenSpam = countsConditionalSpam.get(word);
            float probabilityOfWordGivenSpam = (float) (nWordGivenSpam + alpha) / (float) (spamMessages.size() + alpha*totalVocabulary);
            conditionalSpam.put(word, probabilityOfWordGivenSpam);

            int nWordGivenHam = countsConditionalHam.get(word);
            float probabilityOfWordGivenHam = (float) (nWordGivenHam + alpha) / (float) (hamMessages.size() + alpha*totalVocabulary);
            conditionalHam.put(word, probabilityOfWordGivenHam);
        }
        System.out.println("Successfully fitted model!");
    }


    public String predict(String sentence) {
        String[] wordsInTheSentence = sentence.replaceAll("[^a-zA-Z ]", "").toLowerCase().split("\\s+");
        float pSpamGivenSentence = probabilityOfSpam;
        float pHamGivenSentence = probabilityOfHam;

        for (String word : wordsInTheSentence) {
            if (conditionalSpam.containsKey(word))
                pSpamGivenSentence *= conditionalSpam.get(word);

            if (conditionalHam.containsKey(word))
                pHamGivenSentence *= conditionalHam.get(word);

        }
        return pSpamGivenSentence > pHamGivenSentence ? "spam" : "ham";
    }
}

public class source {
    public static void main(String[] args) throws FileNotFoundException {
        File trainData = new File("C:\\Users\\angel\\IdeaProjects\\186068_Source\\SMSSpamTrain.txt");
        File testData = new File("C:\\Users\\angel\\IdeaProjects\\186068_Source\\SMSSpamTest.txt");
        Scanner readerTrain = new Scanner(trainData);
        Scanner readerTest = new Scanner(testData);

        ArrayList<String> xTrain = new ArrayList<>();
        ArrayList<String> yTrain = new ArrayList<>();
        while (readerTrain.hasNextLine()) {
            String[] data = readerTrain.nextLine().split("\\t");
            yTrain.add(data[0]);
            xTrain.add(data[1]);

        }
        readerTrain.close();

        BayesSpamFilter model = new BayesSpamFilter();
        model.fit(xTrain, yTrain);

        ArrayList<String> xTest = new ArrayList<>();
        ArrayList<String> yTest = new ArrayList<>();
        while (readerTest.hasNextLine()) {
            String[] data = readerTest.nextLine().split("\\t");
            yTest.add(data[0]);
            xTest.add(data[1]);

        }
        readerTest.close();

        int truePositives = 0;
        int trueNegatives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;

        for (int i = 0; i < xTest.size(); i++) {
            String predicted = model.predict(xTest.get(i));
            if (predicted.equals("spam") && yTest.get(i).equals("spam"))
                truePositives++;
            else if (predicted.equals("spam") && yTest.get(i).equals("ham"))
                falsePositives++;
            else if (predicted.equals("ham") && yTest.get(i).equals("ham"))
                trueNegatives++;
            else if (predicted.equals("ham") && yTest.get(i).equals("spam"))
                falseNegatives++;
        }
        System.out.println("True Positives: " + truePositives + "\tFalse Negatives: " + falseNegatives
                + "\nFalse Positives: " + falsePositives + "\tTrue Negatives: " + trueNegatives);
    }
}
