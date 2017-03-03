import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.zip.GZIPInputStream;

import edu.berkeley.nlp.lm.NgramLanguageModel;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.io.IOUtils;
import edu.berkeley.nlp.lm.io.LmReaders;
import edu.berkeley.nlp.lm.util.Logger;

public class LmTests {

	public static void main(String[] args) throws FileNotFoundException, IOException {
		
		boolean isGoogleBinary = false;
		String vocabFile = null;
		String binaryFile = "/home/kukarzev/github/berkeleylm/examples/big_test.binary";
		List<String> files = Collections.singletonList("-");
		
		NgramLanguageModel<String> lm = readBinary(isGoogleBinary, vocabFile, binaryFile);
		computeProb(files, lm);
		
	}

	
	/**
	 * @param files
	 * @param lm
	 * @throws IOException
	 * @throws FileNotFoundException
	 */
	private static double computeProb(List<String> files, NgramLanguageModel<String> lm) throws IOException, FileNotFoundException {
		double logProb = 0.0;
		for (String file : files) {
			Logger.startTrack("Scoring file " + file + "; current log probability is " + logProb);
			final InputStream is = (file.equals("-")) ? System.in : (file.endsWith(".gz") ? new GZIPInputStream(new FileInputStream(file))
				: new FileInputStream(file));
			BufferedReader reader = new BufferedReader(new InputStreamReader(new BufferedInputStream(is)));
			for (String line : Iterators.able(IOUtils.lineIterator(reader))) {
				List<String> words = Arrays.asList(line.trim().split("\\s+"));
				logProb += lm.scoreSentence(words);
			}
			Logger.endTrack();
		}
		return logProb;
	}
	
	
	/**
	 * @param isGoogleBinary
	 * @param vocabFile
	 * @param binaryFile
	 * @return
	 */
	private static NgramLanguageModel<String> readBinary(boolean isGoogleBinary, String vocabFile, String binaryFile) {
		NgramLanguageModel<String> lm = null;
		if (isGoogleBinary) {
			Logger.startTrack("Reading Google Binary " + binaryFile + " with vocab " + vocabFile);
			lm = LmReaders.readGoogleLmBinary(binaryFile, vocabFile);
			Logger.endTrack();
		} else {
			Logger.startTrack("Reading LM Binary " + binaryFile);
			lm = LmReaders.readLmBinary(binaryFile);
			Logger.endTrack();
		}
		return lm;
	}

}
