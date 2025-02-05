import os
import sys
import json
import boto3
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
from collections import Counter
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Set environment variables for writable directories
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["NUMBA_CACHE_DIR"] = "/tmp"
nltk.data.path.append("/tmp")

# Ensure necessary nltk downloads
nltk.download('vader_lexicon', download_dir='/tmp')
nltk.download('punkt', download_dir='/tmp')

# AWS Clients
dynamodb = boto3.client('dynamodb', region_name='ap-south-1')
s3 = boto3.client('s3', region_name='ap-south-1')

def get_audio_length(audio_file_path):
    """Calculate the length of an audio file."""
    y, sr = librosa.load(audio_file_path, sr=None)
    length_in_seconds = librosa.get_duration(y=y, sr=sr)
    return length_in_seconds

def calculate_clarity(audio_file_path):
    """Calculate audio clarity score using librosa."""
    y, sr = librosa.load(audio_file_path, sr=None)
    S = np.abs(librosa.stft(y))
    clarity_score = np.mean(librosa.feature.spectral_flatness(S=S))
    return clarity_score

def transcribe_audio(audio_file_path):
    """Transcribe the audio file using Hugging Face Wav2Vec2."""
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    waveform, sample_rate = torchaudio.load(audio_file_path)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()

def calculate_sentiment_score(text):
    """Calculate sentiment score of the transcribed text using nltk."""
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    
    # Remove "compound" and rename keys
    sentiment_scores.pop("compound", None)
    sentiment_scores["positive"] = sentiment_scores.pop("pos")
    sentiment_scores["negative"] = sentiment_scores.pop("neg")
    sentiment_scores["neutral"] = sentiment_scores.pop("neu")
    
    return sentiment_scores

def convert_mp4_to_wav(mp4_file_path):
    """Convert MP4 file to WAV format using torchaudio."""
    waveform, sample_rate = torchaudio.load(mp4_file_path)
    
    # Save the waveform as a WAV file
    wav_file_path = "/tmp/temp.wav"
    torchaudio.save(wav_file_path, waveform, sample_rate)
    
    return wav_file_path

def audio_analysis(audio_file_path):
    # Convert MP4 to WAV if necessary
    if audio_file_path.lower().endswith('.mp4'):
        audio_file_path = convert_mp4_to_wav(audio_file_path)
    
    # Calculate audio length
    audio_length = get_audio_length(audio_file_path)
    
    # Calculate audio clarity
    audio_clarity = calculate_clarity(audio_file_path)
    
    # Transcribe audio
    transcribed_text = transcribe_audio(audio_file_path)
    
    # Calculate sentiment score
    sentiment_score = calculate_sentiment_score(transcribed_text)

    # Determine overall sentiment based on the maximum value in sentiment_score
    overall_sentiment = max(sentiment_score, key=sentiment_score.get)

    # Convert all values to JSON serializable types
    return json.dumps({
        "audio_length": float(audio_length),
        "audio_clarity": float(audio_clarity),
        "transcribed_text": transcribed_text,
        "sentiment_score": {k: float(v) for k, v in sentiment_score.items()},
        "overall_sentiment": overall_sentiment
    })

def lambda_handler(event, context):
    """Lambda function entry point."""
    try:
        id = event['id']
        print("Processing ID:", id)
        # Fetch data from DynamoDB
        response = dynamodb.get_item(
            TableName='SurveyFormDetailsUser',
            Key={'id': {'S': id}}
        )
        item = response.get('Item')
        print("DynamoDB item:", item)
        if not item:
            return {'statusCode': 404, 'body': json.dumps({'message': 'ID not found'})}
        if item['status']['S'] != 'submitted':
            return {'statusCode': 400, 'body': json.dumps({'message': 'Status not submitted'})}
        form_response_str = item.get('formResponse', {}).get('S')
        if not form_response_str:
            return {'statusCode': 404, 'body': json.dumps({'message': 'formResponse not found or invalid format'})}
        form_response = json.loads(form_response_str)
        file_names = form_response.get('file_names', [])
        if not file_names:
            return {'statusCode': 404, 'body': json.dumps({'message': 'file_names not found in formResponse'})}
        
        analysis_results = {}
        file_responses = []
        
        for file_name in file_names:
            # Validate file extension
            if not file_name.lower().endswith(('.mp3', '.wav')):  
                file_responses.append({
                    'file_name': file_name,
                    'statusCode': 400,
                    'message': f'File {file_name} is not an audio file'
                })
                continue
            # Check if file exists in S3
            try:
                s3.head_object(Bucket='surveytoolptag', Key=f'SubmittedForms/{file_name}')
            except:
                file_responses.append({
                    'file_name': file_name,
                    'statusCode': 404,
                    'message': f'File {file_name} not found in S3'
                })
                continue
            # Download file from S3
            download_path = f'/tmp/{file_name}'
            s3.download_file('surveytoolptag', f'SubmittedForms/{file_name}', download_path)
            # Analyze audio
            analysis_result = audio_analysis(download_path)
            print("Audio analysis result:", analysis_result)
            if analysis_result is None:
                file_responses.append({
                    'file_name': file_name,
                    'statusCode': 500,
                    'message': f'Error in audio analysis for file {file_name}'
                })
                continue
            analysis_results[file_name] = json.loads(analysis_result)
            # Clean up
            os.remove(download_path)
            
            file_responses.append({
                'file_name': file_name,
                'statusCode': 200,
                'message': 'Audio analysis completed'
            })
        
        # Check if analysis_results is empty and set it to null if it is
        if not analysis_results:
            analysis_results = None
        
        # Update DynamoDB
        dynamodb.update_item(
            TableName='SurveyAnalysisData',
            Key={'id': {'S': id}},
            UpdateExpression="SET audio_kpi = :val",
            ExpressionAttributeValues={':val': {'S': json.dumps(analysis_results, ensure_ascii=False)}}
        )
        
        return {
            'statusCode': 200,
            'body': {
                'message': 'Audio analysis completed',
                'file_responses': file_responses
            }
        }
    except Exception as e:
        print(f"Error in Lambda function: {e}")
        return {'statusCode': 500, 'body': json.dumps({'message': str(e)})}