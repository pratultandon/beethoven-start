import midi
import numpy as np
import glob  
from tqdm import tqdm

lowerBound = 21 #The lowest note
upperBound = 105 #The highest note
span = upperBound-lowerBound #The note range
num_timesteps      = 5 #The number of note timesteps that we produce with each RBM


def write_song(path, song):
    #Reshape the song into a format that midi_manipulation can understand, and then write the song to disk
    song = np.reshape(song, (song.shape[0]*num_timesteps, 3*span))
    noteStateMatrixToMidi(song, name=path)

def get_song(path):
    #Load the song and reshape it to place multiple timesteps next to each other
    song = np.array(midiToNoteStateMatrix(path))
    song = song[:np.floor(song.shape[0]/num_timesteps).astype(np.int32)*num_timesteps]
    song = np.reshape(song, [song.shape[0]/num_timesteps, song.shape[1]*num_timesteps])
    return song

def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = get_song(f)
            if np.array(song).shape[0] > 50/num_timesteps:
                songs.append(song)
        except Exception as e:
            print f, e            
    return songs

def midiToNoteStateMatrix(midifile, squash=True, span=span):
    pattern = midi.read_midifile(midifile)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    statematrix = []
    time = 0

    state = [[0,0,0] for x in range(span)]
    statematrix.append(state)
    condition = True
    while condition:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0,oldstate[x][2]] for x in range(span)]
            statematrix.append(state)
        for i in range(len(timeleft)): #For each track
            if not condition:
                break
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                        print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            if evt.velocity < 50:
                                state[evt.pitch-lowerBound] = [0, 0, 0]
                            else:
                                state[evt.pitch-lowerBound] = [0, 0, 1]
                        else:
                            if evt.velocity < 50:
                                state[evt.pitch-lowerBound] = [1, 1, 0]
                            else:
                                state[evt.pitch-lowerBound] = [1, 1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
										2 + 2
                    # if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        # print "Found time signature event {}. Bailing!".format(evt)
                        # out =  statematrix
                        # condition = False
                        # break
                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    S = np.array(statematrix)
    statematrix = np.hstack((S[:, :, 0], S[:, :, 1], S[:, :, 2]))
    statematrix = np.asarray(statematrix).tolist()
    return statematrix

def noteStateMatrixToMidi(statematrix, name="example", span=span):
    statematrix = np.array(statematrix)
    print statematrix, len(statematrix.shape), statematrix.shape, span
    if not len(statematrix.shape) == 3:
        statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:2*span], statematrix[:, 2*span:]))
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    span = upperBound-lowerBound
    tickscale = 55
    
    lastcmdtime = 0
    prevstate = [[0,0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if (p[0])== 1:
                if (n[0]) == 0:
                    offNotes.append((i, n[2]))
                elif (n[1]) == 1:
                    offNotes.append((i, p[2]))
                    onNotes.append((i, n[2]))
            elif (n[0]) == 1:
                onNotes.append((i, n[2]))
        print onNotes
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, velocity = (int(note[1]*50)+45)%128, pitch=note[0]+lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=(int(note[1]*50)+45)%128, pitch=note[0]+lowerBound))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)
