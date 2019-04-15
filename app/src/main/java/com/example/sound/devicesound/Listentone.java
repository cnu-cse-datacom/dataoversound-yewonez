package com.example.sound.devicesound;
/*
import com.google.zxing.common.reedsolomon.GenericGF;
import com.google.zxing.common.reedsolomon.ReedSolomonDecoder;
import com.google.zxing.common.reedsolomon.ReedSolomonException;
*/
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.support.annotation.RequiresApi;
import android.util.Log;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.*;
/*
import java.nio.charset.StandardCharsets;
*/
import java.util.ArrayList;

import static java.lang.Math.*;

public class Listentone {

    int HANDSHAKE_START_HZ = 4096;
    int HANDSHAKE_END_HZ = 5120 + 1024;

    int START_HZ = 1024;
    int STEP_HZ = 256;
    int BITS = 4;

    int FEC_BYTES = 4;

    private int mAudioSource = MediaRecorder.AudioSource.MIC;
    private int mSampleRate = 44100;
    private int mChannelCount = AudioFormat.CHANNEL_IN_MONO;
    private int mAudioFormat = AudioFormat.ENCODING_PCM_16BIT;
    private float interval = 0.1f;

    private int mBufferSize = AudioRecord.getMinBufferSize(mSampleRate, mChannelCount, mAudioFormat);

    public AudioRecord mAudioRecord = null;
    int audioEncodig;
    boolean startFlag;
    FastFourierTransformer transform;


    public Listentone(){
        transform = new FastFourierTransformer(DftNormalization.STANDARD);
        startFlag = false;
        mAudioRecord = new AudioRecord(mAudioSource, mSampleRate, mChannelCount, mAudioFormat, mBufferSize);
        mAudioRecord.startRecording();
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    public void PreRequest(){
        //decode.py의 linsten_linux
        //Log.d("TAG", "Message"); TEST
        //fftfreq(100, 1.0 );
        //fftfreq(101, 1.0);

        /*이 내용은 Listentone class의 private변수로 선언 되어 있음*/
        /*mic = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device="hw:1")
        mic.setchannels(1)
        mic.setrate(44100)
        mic.setformat(alsaaudio.PCM_FORMAT_S16_LE)*/

        //in_packet = False     -> startFlag, Listentone 클래스의 변수.
        ArrayList<Double> packet = new ArrayList<>();            //packet = []
        ArrayList<Integer> byte_stream = new ArrayList<>();      //while문 안에 byte_stream

        int blocksize = findPowerSize((int)(long)Math.round(interval/2*mSampleRate));
        short[] buffer = new short[blocksize];
        //num_frames = int(round((interval / 2) * frame_rate))

        while (true) {

            int bufferedReadResult = mAudioRecord.read(buffer, 0, blocksize);
            double[] chunk = new double[blocksize];
            //  AudioRecord의 read 함수를 통해 pcm data 를 읽어옴     l, data = mic.read()
        /*
        -3 : ERROR_INVALID_OPERATION if the object isn't properly initialized
        -2 : ERROR_BAD_VALUE if the parameters don't resolve to valid data and indexes
        -6 : ERROR_DEAD_OBJECT if the object is not valid anymore and needs to be recreated. The dead object error code is not returned if some data was successfully transferred. In this case, the error is returned at the next read()
        -1 : ERROR in case of other error
        */
            if(bufferedReadResult<0){       //if not l: continue
                continue;
            }
            //chunk =np.fromstring(data, dtype = np.int16)
            for(int i = 0 ;i< blocksize ; i++){
                chunk[i] = buffer[i];
            }
            double dom = findFrequency(chunk);      // dom = dominant(frame_rate, chunk)
            if (startFlag  && match(dom, HANDSHAKE_END_HZ)){
                //Log.d("END_HZ","enter");
                byte_stream = extract_packet(packet);
                Log.d("Byte_TEST", "Byte:" + byte_stream.toString());
                int byte_stream_size= byte_stream.size();
                int[] received = new int[byte_stream_size];
//                byte[] b = new byte[byte_stream_size-4];  //test
//                for(int i = 0; i< byte_stream_size-4; i++){
//                    b[i]  = (byte)received[i];
//                }
//                String s = new String(b, StandardCharsets.US_ASCII);
//                Log.d("Byte_TEST1", "output: " + s);
                for(int i = 0; i< byte_stream_size; i++){
                    received[i] = byte_stream.get(i);
                }
                /*try {
                    //byte_stream = RSCodec(FEC_BYTES).decode(byte_stream)
                    //byte_stream = byte_stream.decode("utf-8")
                    //int temp = received.length;
                    //Log.d("received size", "rsize : "+ temp);
                    //GenericGF RS_PARAM = new GenericGF(0x20, temp, 1);
                    ReedSolomonDecoder rsdecode = new ReedSolomonDecoder(GenericGF.DATA_MATRIX_FIELD_256);      //GenericGF값을 어떻게 넣어야 되는지 모르겠음. 유한체가 뭐야 하....
                    rsdecode.decode(received, 4);
                    Log.d("Byte_TEST2", "Byte:" + byte_stream.toString());

                    for(int i = 0; i<received.length; i++){
                        Log.d("Byte_OUT", ":" + received[i]);
                    }
                }catch (ReedSolomonException e){
                    Log.d("ReedSolomonError","ERROR OCURED");
                    packet.clear();     //packet = []
                    byte_stream.clear();
                    received = null;
                    startFlag = false;  //in_packet = False
                    continue;   //pass
                }*/
                StringBuffer stb= new StringBuffer();

                for(int i =0 ;i < byte_stream.size(); i++){
                    stb.append(byte_stream.get(i));
                }
                packet.clear();     //packet = []
                byte_stream.clear();
                received = null;
                startFlag = false;  //in_packet = False
            }
            else if (startFlag) {
                packet.add(dom);
                //Log.d("startFlag at add","value : " + startFlag);
                Log.d("Dominant","value : " + dom);
            } else if (match(dom, HANDSHAKE_START_HZ)) {
                startFlag  = true;
                //Log.d("startFlag","value : " + startFlag);
            }
        }

    }

    private double findFrequency(double[] toTransform) {   //이전 decode.py의 dominant 역할을 함
        int len = toTransform.length;
        double[] real = new double[len];
        double[] img = new double[len];
        double realNum;
        double imgNum;
        double[] mag = new double[len];


        Complex[] complx = transform.transform(toTransform,TransformType.FORWARD);
        Double[] freqs = this.fftfreq(complx.length, 1);         //freqs = np.fft.fftfreq(len(chunk))

        for(int i = 0; i< complx.length; i++ ) {                    //decode.py에서 w = np.fft.fft(chunk)
        realNum = complx[i].getReal();
        imgNum = complx[i].getImaginary();
        mag[i] = Math.sqrt((realNum * realNum) + (imgNum * imgNum));    //(np.abs(w))
        }
        double temp = 0;
        int peak_coeff=0;
        for(int i =0; i< mag.length; i++){
            if(temp < mag[i]){
                temp = mag[i];
            }
        }
        for(int i = 0; i< mag.length; i++){      //peak_coeff = np.argmax(np.abs(w)), 복소수에서 최대값의 인덱스값을 구함.
            if(temp == mag[i]){
                peak_coeff = i;
                break;
            }
        }
        double peak_freq = freqs[peak_coeff];
        //Log.d("TEST-peak_freq", "peak_freq : " + abs(peak_freq * mSampleRate));
        return abs(peak_freq * mSampleRate);
    }

    private boolean match(double freq1, double freq2){   //decode.py의 match def
        return abs(freq1-freq2) < 20;
    }

    private Double[] fftfreq(int n, double d){
        double val = 1.0 / (n * d);     //double val = 1.0 / (n * d)
        int[] results = new int[n];     //results = empty(n, int)
        int N = (n-1) / 2 + 1;          //N = (n-1)//2 + 1
        for(int i = 0; i<N; i++){      // results[:N] = p1 = arange(0, N, dtype=int)
            results[i] = i;
        }
        int p2 = -(n/2);                //p2 = arange(-(n//2), 0, dtype=int)
        for(int i = N; i<n; i++){     // results[N:] = p2
            results[i] = p2;
            ++p2;
        }
        Double[] return_Double_Array = new Double[n];
        for(int i = 0; i<n; i++){
            return_Double_Array[i] = results[i] * val;
            //Log.d("TAG: fftfreq"+ n, "Message: "+ return_Double_Array[i]);
        }
        return return_Double_Array;
    }

    private int findPowerSize(int R_value){
        if(R_value == 0) return 0;
        int a = 1;
        while(R_value > a){
            a = a*2;
        }
        Log.d("TEST-findPowerSize : ", "a : " + a + "," + R_value);
            return a;
    }

    private ArrayList<Integer> extract_packet(ArrayList<Double> freqs) { // extract_packet 그대로 들고 왔음.
        ArrayList<Integer> bit_chunks = new ArrayList<>();      //요놈이 bit_chunks
        ArrayList<Integer> bit_chunks2 = new ArrayList<>();      //요놈이 bit_chunks22
        ArrayList<Double> freqs2 = new ArrayList<>();           //freqs[::2] 이거 할당할꺼

        for(int i = 0; i< freqs.size(); i++){      //https://blog.wonkyunglee.io/3 참고
            //freqs = freqs[::2]
            //원래는 i++이 아니라 i+=2해야 되는데 입력되는 주파수 살펴보니까
            //2개씩들어오는 게 아니고 한개씩 들어오드라
            //흐헝
            freqs2.add(freqs.get(i));
        }
        for(int i = 0; i< freqs2.size(); i++){      //bit_chunks = [int(round((f - START_HZ) / STEP_HZ)) for f in freqs]
            bit_chunks.add((int)Math.round((freqs2.get(i)-START_HZ) / STEP_HZ));
        }
        for(int i = 1; i< bit_chunks.size(); i++){  //bit_chunks = [c for c in bit_chunks[1:] if 0 <= c < (2 ** BITS)]
            int c = bit_chunks.get(i);
            if((c >= 0) && (c < pow(2,BITS))){
                bit_chunks2.add(c);
            }
        }
        return decode_bitchunks(BITS, bit_chunks2);

    }

    private ArrayList<Integer> decode_bitchunks(int chunk_bits, ArrayList<Integer> chunks) { // decode_bitchunks, 그대로 들고 왔어요
        ArrayList<Integer> out_bytes = new ArrayList<>();
        int next_read_chunk = 0;
        int next_read_bit = 0;
        int Shift_byte = 0;
        int bits_left = 8;

        while(next_read_chunk < chunks.size()) {
            int can_fill = chunk_bits - next_read_bit;
            int to_fill = min(bits_left, can_fill);
            int offset = chunk_bits - next_read_bit - to_fill;
            Shift_byte <<= to_fill;
            int shifted = chunks.get(next_read_chunk) & (((1 << to_fill) - 1) << offset);
            Shift_byte |= shifted >> offset;
            bits_left -= to_fill;
            next_read_bit += to_fill;
            if (bits_left <= 0) {
                out_bytes.add(Shift_byte);
                Shift_byte = 0;
                bits_left = 8;
            }
            if (next_read_bit >= chunk_bits) {
                next_read_chunk += 1;
                next_read_bit -= chunk_bits;
            }
        }
        //Log.d("out_bytes","output : "+out_bytes.toString());
        return out_bytes;
    }

}
