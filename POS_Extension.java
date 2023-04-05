import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Hidden Markov Model on Brown Corpus sentences for identifying Part-of-Speech Tags
 *
 * @author William Dinauer and Erich Woo, Dartmouth CS 10, Winter 2020
 */
public class POS_Extension {

    private Map<String, Map<String, Double>> transitions;
    private Map<String, Map<String, Double>> observations;

    private String trainS;
    private String trainT;

    private double U = -16;  //test around with unseen weight constant for optimal results

    /**
     * Constructor for Hidden Markov Model object
     *
     * @param trainS the training sentence path
     * @param trainT the training tags path
     */
    public PS5_EC(String trainS, String trainT) {
        transitions = new HashMap<String, Map<String, Double>>();      //tag -> tag -> log(#/total)
        observations = new HashMap<String, Map<String, Double>>();     //tag -> words -> log(#/total)
        setTrainS(trainS);
        setTrainT(trainT);
    }

    /**
     * Sets U value
     * @param u the given u value
     */
    public void setU(double u) {
        this.U = u;
    }

    /**
     * Sets the training sentence
     * @param trainS the training sentence filename path
     */
    public void setTrainS(String trainS) {
        this.trainS = trainS;
    }

    /**
     * Sets the training tags
     * @param trainT the training tags filename path
     */
    public void setTrainT(String trainT) {
        this.trainT = trainT;
    }

    /**
     * Increments the count of a specific key-subkey pairing in a given map
     * @param map the map to increment count from
     * @param key the outer key
     * @param subkey the inner key
     */
    public static void add(Map<String, Map<String, Double>> map, String key, String subkey) {
        if (!map.containsKey(key)) { map.put(key, new HashMap<String,Double>()); } // if key not in map, create key-map pair
        if (!map.get(key).containsKey(subkey)) { map.get(key).put(subkey, 0.0); } // if subkey not in submap, create subkey-value pair
        map.get(key).put(subkey, map.get(key).get(subkey)+1);   //increment count of specific subkey instance by 1
    }

    /**
     * Replaces the count of a key-subkey pairing with the ln(count/total) value
     * @param map the map to replace from
     */
    public static void log(Map<String, Map<String, Double>> map) {
        for (String key : map.keySet()) {  // for each key in map
            int sum = 0;

            //sum up all double values for a specific key ("normalized by" value)
            for (String subkey : map.get(key).keySet()) {
                sum += map.get(key).get(subkey);
            }

            //replace count value with ln (probability)
            for (String subkey: map.get(key).keySet()) {
                double probability = map.get(key).get(subkey)/sum;
                map.get(key).put(subkey, Math.log(probability));
            }
        }
    }

    /**
     * Trains the given model
     * @param model the model to train on
     * @throws IOException for invalid file on BufferedReader
     */
    public static void trainModel(PS5_EC model) throws IOException {
        BufferedReader sentences = new BufferedReader(new FileReader(model.trainS));
        BufferedReader tag = new BufferedReader(new FileReader(model.trainT));
        String sLine;       // current read line of sentences text file
        String tLine;       // current read line of tags text file

        while ((sLine = sentences.readLine())!=null){
            sLine.toLowerCase();
            tLine = tag.readLine();
            String[] wordArray = sLine.split(" ");  // build array of words from white space separation
            String[] tagArray = tLine.split(" ");   // build array of tags from white space separation

            String start = "#";
            String currWord;
            String currTag = start;
            for (int x = 0; x < wordArray.length; x++){  // changed from x = -1 to 0
                //add to transitions map; because of this, "." will never be added as currTag to transitions map
                String nextTag = tagArray[x]; //temporarily, next tag is the first tag in array for # -> first
                add(model.transitions, currTag, nextTag);

                //add to tagToWord map
                currWord = wordArray[x];
                currTag = tagArray[x];   //reset current tag to actual first index at 0
                add(model.observations, currTag, currWord);
            }
        }
        sentences.close();
        tag.close();

        log(model.transitions);
        log(model.observations);
    }

    /**
     * Tests the given model using Viterbi
     * @param model the model to test on
     * @throws IOException for invalid file on BufferedReader
     */
    public static void testModel(PS5_EC model, String testSentences, String testTags) throws IOException{
        BufferedReader testSent = new BufferedReader(new FileReader(testSentences));
        BufferedReader testTag = new BufferedReader(new FileReader(testTags));

        List<String> actualTags = new ArrayList<String>();
        List<String> guessTags = new ArrayList<String>();

        String sLine;
        String tLine;
        while ((sLine = testSent.readLine())!=null) {
            sLine.toLowerCase();
            tLine = testTag.readLine();
            String[] tagArray = tLine.split(" ");
            for (int i = 0; i < tagArray.length; i++) {
                actualTags.add(tagArray[i]);
            }

            List<String> lineTags = viterbi(model, sLine);
            for (String tag : lineTags) {
                guessTags.add(tag);
            }
        }
        int same = 0;
        int diff = 0;

        if (actualTags.size() == guessTags.size()) {
            for (int i = 0; i < actualTags.size(); i++) {
                if (actualTags.get(i).equals(guessTags.get(i))) {
                    same++;
                } else {
                    diff++;
                }
            }
        }
        System.out.println("tags correct: " + same + "\ntags incorrect: " + diff);
        System.out.println("Actual tags: " + actualTags);
        System.out.println("Model's tags: " + guessTags);
        System.out.println();
    }

    /**
     * Do Viterbi Decoding on a line using trained model
     * @param model the trained model
     * @param sLine the line to tag
     * @return returns a list of tags matching each word of the line given
     */
    public static List<String> viterbi(PS5_EC model, String sLine) {
        sLine.toLowerCase();
        String[] wordArray = sLine.split(" ");
        List<String> lineTags = new ArrayList<String>();

        Set<String> currStates = new HashSet<String>();                                   // set of all possible current states
        Map<String, Double> currScores = new HashMap<String, Double>();                   // map (currState -> nextScore)
        ArrayList<Map<String, String>> backtracer = new ArrayList<Map<String, String>>(); // size of #testWords, each element is a map of winners for each possible cs <- ns pairing

        currStates.add("#");
        currScores.put("#", 0.0);

        for (int x = 0; x < wordArray.length; x++) {                        // for each word in test sentence
            Map<String, String> data = new HashMap<String, String>();       // next, curr
            Set<String> nextStates = new HashSet<String>();                 // set of all possible next states
            Map<String, Double> nextScores = new HashMap<String, Double>(); // per word: for each nextState tag, choose best currentState to backtrack to

            String word = wordArray[x];
            //nested for loop of every tag -> tag pairing possible
            for (String cs : currStates) {                                  //cs = specific current state (a tag)
                if (model.transitions.containsKey(cs)) {                    // if cs is "." , skip over since that is end of line
                    for (String ns : model.transitions.get(cs).keySet()) {  //ns = specific next state (a tag)
                        double obs = model.U;                               //instantiate observation score to unseen(U) value
                        nextStates.add(ns);                                 //won't add again if next state is already added

                        //if the word is observed to have this tag in the training files, change obs to its ln value
                        if (model.observations.get(ns).containsKey(word)) {
                            obs = model.observations.get(ns).get(word);
                        }

                        double nextScore = currScores.get(cs) + model.transitions.get(cs).get(ns) + obs;  //current + transition + observation

                        if (!nextScores.containsKey(ns) || nextScore > nextScores.get(ns)) {  // ns not nextScore
                            nextScores.put(ns, nextScore); // put (or replace) with new best nextScore
                            data.put(ns, cs);   // put (or replace) with new best winner for (cs <- ns) pairing
                        }
                    }
                }
            }
            backtracer.add(data);
            currStates = nextStates;
            currScores = nextScores;

            Stack<String> tagHolder = new Stack<String>();
            //select the biggest winner for backtracing when finished with last Viterbi-ing of last word in line
            if (x == wordArray.length - 1) {
                //choose highest nextScore in nextScores Map
                double bestScore = 0;
                String bestTag = "";

                for (String ns: nextScores.keySet()) {
                    if (bestScore == 0 || nextScores.get(ns) > bestScore) {
                        bestScore = nextScores.get(ns);
                        bestTag = ns;
                    }
                }
                tagHolder.push(bestTag);                      //add the last tag to backtrace from
                for (int i = backtracer.size() - 1; i > 0; i--) {
                    bestTag = backtracer.get(i).get(bestTag); //backtrace temp tag all the way to first word -1 (dont want start # sign)
                    tagHolder.push(bestTag);
                }
            }
            while (!tagHolder.isEmpty()) {
                lineTags.add(tagHolder.pop());
            }
        }
        return lineTags;
    }

    /**
     * Console-based test method, matching tags to a user-inputted sentence
     *
     * @param model the trained model
     */
    public static void userTest(PS5_EC model) {
        Scanner input = new Scanner(System.in);
        boolean playing = true;

        System.out.println("Welcome to the User-Interactive Sentence to Tag Tester!\nYou give us a sentence, we'll tell you its tags\n");
        while (playing) {
            System.out.println("Enter a sentence: ");
            String sentence = "";
            if (input.hasNextLine()) {
                sentence = input.nextLine();
            }
            List<String> tags = viterbi(model, sentence);
            String[] sentenceArray = sentence.split(" ");
            for (int i = 0; i < sentenceArray.length; i++) {
                sentenceArray[i] += "/" + tags.get(i);
            }
            for (String word : sentenceArray) {
                System.out.print(word + " ");
            }
            System.out.println("\n");
            System.out.println("Play again? (y/n): ");
            String answer = input.nextLine();
            if (!answer.toLowerCase().equals("y") && !answer.equals("yes")) {
                playing = false;
            }
        }
    }

    /**
     * A simple test based on PD-7
     * @param model the given model to put hard-coded transitions and observations on
     */
    public static void simpleTest(PS5_EC model) {
        Map<String, Map<String, Double>> transitions = new HashMap<String, Map<String, Double>>();
        Map<String, Map<String, Double>> observations = new HashMap<String, Map<String, Double>>();

        Map<String, Double> map = new HashMap<String, Double>();
        map.put("NP", 3.0);
        map.put("N", 7.0);
        transitions.put("#", map);
        map = new HashMap<String, Double>();
        map.put("CNJ", 2.0);
        map.put("V", 8.0);
        transitions.put("NP", map);
        map = new HashMap<String, Double>();
        map.put("NP", 2.0);
        map.put("N", 4.0);
        map.put("V", 4.0);
        transitions.put("CNJ", map);
        map = new HashMap<String, Double>();
        map.put("V", 8.0);
        map.put("CNJ", 2.0);
        transitions.put("N", map);
        map = new HashMap<String, Double>();
        map.put("N", 4.0);
        map.put("CNJ", 2.0);
        map.put("NP", 4.0);
        transitions.put("V", map);
        model.transitions = transitions;

        map = new HashMap<String, Double>();
        map.put("cat", 4.0);
        map.put("dog", 4.0);
        map.put("watch", 2.0);
        observations.put("N", map);
        map = new HashMap<String, Double>();
        map.put("and", 10.0);
        observations.put("CNJ", map);
        map = new HashMap<String, Double>();
        map.put("get", 1.0);
        map.put("watch", 6.0);
        map.put("chase", 3.0);
        observations.put("V", map);
        map = new HashMap<String, Double>();
        map.put("chase", 10.0);
        observations.put("NP", map);
        model.observations = observations;

        model.setU(-10);

        //test 1
        String sentence = "chase watch dog chase watch";
        ArrayList<String> expectedTags = new ArrayList<String>();
        expectedTags.add("NP");
        expectedTags.add("V");
        expectedTags.add("N");
        expectedTags.add("V");
        expectedTags.add("N");
        if (simpleTestHelp(model, sentence, expectedTags))
            System.out.println("Test 1 passed!");
        else
            System.out.println("Failed test 1");
        System.out.println();

        //test 2
        expectedTags = new ArrayList<String>();
        sentence = "cat and dog chase and chase";
        expectedTags.add("N");
        expectedTags.add("CNJ");
        expectedTags.add("N");
        expectedTags.add("V");
        expectedTags.add("CNJ");
        expectedTags.add("NP");
        if (simpleTestHelp(model, sentence, expectedTags))
            System.out.println("Test 2 passed!");
        else
            System.out.println("Failed test 2");
        System.out.println();
    }

    /**
     * Tests whether or not the viterbi decoder passes the expected result
     * @param model the model to test on
     * @param sentence  the sentence being tested
     * @param expected  a list of the expected tags for the sentence
     * @return
     */
    public static boolean simpleTestHelp(PS5_EC model, String sentence, ArrayList<String> expected){
        int x=0;
        List<String> guess = viterbi(model, sentence);      //calls viterbi decoding
        System.out.println("Sentence: " + sentence);
        System.out.println("Actual Tags: "+expected);                       //prints expected tags
        System.out.println("Model's Tags: "+guess);                          //prints tags received from viterbi (for visual comparison)
        for (String tag: guess){                            //compares expected and the guess
            if (!tag.equals(expected.get(x)))
                return false;                               //for any difference return false;
            x++;
        }
        return true;                                        //if there are no differences, return true
    }

    /**
     * Method for EC. Returns a random sentence that could be produced by a trained model.
     * @param model the model to test on
     * @return a random sentence
     */
    public static String randomSentence(PS5_EC model){
        String curr = "DET";
        String word = getRandomWord(model, curr);
        ArrayList<String> words = new ArrayList<String>();
        ArrayList<String> tags = new ArrayList<String>();
        words.add(word);
        tags.add(curr);
        int count = 1;
        while (model.transitions.containsKey(curr)){
            Set<String> goodTags = new HashSet<String>();
            for (String tag: model.transitions.get(curr).keySet()){
                if (model.transitions.get(curr).get(tag)>-5){
                    goodTags.add(tag);
                }
            }
            String randWord = "";
            String randTag = "";
            while (randWord.length()<2) {
                randTag = getRandomTag(goodTags);
                randWord = getRandomWord(model, randTag);
            }
            words.add(randWord);
            tags.add(randTag);
            count++;
            if (count>8)
                curr = "STOP";
            else
                curr = randTag;
        }
        String result = words.get(0);
        for (int i = 1; i<words.size()-1; i++){
            result+=" "+ words.get(i);
        }
        result+=".";
        return result;
    }

    /**
     * Returns a random tag of all tags in a model
     * @param tags set of good possible tags
     * @return a random tag
     */
    public static String getRandomTag(Set<String> tags){
        int steps = (int)(Math.random()*tags.size());
        int x = 0;
        String tag = "";
        for (String t : tags){
            if (x==steps) {
                tag = t;
                break;
            }
            x++;
        }
        return tag;
    }
    /**
     * Returns a random word given a tag
     * @param model the model to test on
     * @return a random word
     */
    public static String getRandomWord(PS5_EC model, String tag){
        int steps = (int)(Math.random()*model.observations.get(tag).size());
        int x = 0;
        String result = "";
        for (String word: model.observations.get(tag).keySet()){
            if (x==steps) {
                result = word;
                break;
            }
            x++;
        }
        return result;
    }
    /**
     * Main method to execute PS5_EC exercises including tests
     */
    public static void main(String[] args) {
        //Test with hardcoded graph based on PD-7
        PS5_EC testModel = new PS5_EC("","");
        simpleTest(testModel);

        //Train and test on simple text sets
        System.out.println("Simple:");
        PS5_EC modelS = new PS5_EC("inputs/simple-train-sentences.txt","inputs/simple-train-tags.txt");
        try {
            trainModel(modelS);
            testModel(modelS,"inputs/simple-test-sentences.txt","inputs/simple-test-tags.txt");
        }
        catch (Exception e) {
            System.out.println("Invalid text files");
        }

        //Train and test on brown text sets
        System.out.println("Brown:");
        PS5_EC modelB = new PS5_EC("inputs/brown-train-sentences.txt","inputs/brown-train-tags.txt");
        try {
            trainModel(modelB);
            testModel(modelB,"inputs/brown-test-sentences.txt","inputs/brown-test-tags.txt");
        }
        catch (Exception e) {
            System.out.println("Invalid text files");
        }
        System.out.println("Random Sentence: " + randomSentence(modelB) + "\n");
        //console-based test using a brown-trained model
        userTest(modelB);
    }
}