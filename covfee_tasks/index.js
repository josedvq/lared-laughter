// import {
//     VideoInstructionsTask, 
//     LocalAudioInstructionsTask, 
//     LocalVideoInstructionsTask, 
//     LocalAudiovisualInstructionsTask
// } from './video_instructions'

import {
    GeneralInstructionsTask,
    VideoInstructionsTask, 
    AudioInstructionsTask, 
    AVInstructionsTask
} from './pilot_instructions'

export default { 
    'GeneralInstructionsTask': GeneralInstructionsTask,
    'VideoInstructionsTask': VideoInstructionsTask,
    'AudioInstructionsTask': AudioInstructionsTask,
    'AVInstructionsTask': AVInstructionsTask
}