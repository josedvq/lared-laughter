import * as React from 'react'
import {
    Row,
    Col,
    Typography,
    Timeline,
    Button,
    Input,
    Alert
} from 'antd';
const { Title, Paragraph, Text } = Typography

class GeneralInstructionsTask extends React.Component {
    render() {
        return <>
            <Row gutter={16} style={{ 'marginTop': '20px' }}>
                <Col span={22} offset={1}>
                    <Title></Title>
                    <Paragraph>
                        This task consists in watching a series videos of approximately 10 seconds each, and rating the requested quantities (perceived level of arousal, valence and involvement).
                    </Paragraph>

                    <Paragraph>
                        Pay attention to the movements of the person and rate the enjoyment according to your own perception of how much they are enjoying their conversation.
                    </Paragraph>

                    <Paragraph>
                        For each video, the process is the following:
                    </Paragraph>
                    <Timeline>
                        <Timeline.Item>The video will start with a frame showing the target person enclosed in a bounding box. Make sure to find the target person before watching the video.</Timeline.Item>
                        <Timeline.Item>Play the video, paying attention to the target person.</Timeline.Item>
                        <Timeline.Item>Answer the questions about how much the person is enjoying their conversation.</Timeline.Item>
                        <Timeline.Item>Click next to go to the next segment.</Timeline.Item>
                    </Timeline>
                </Col>
            </Row>
            <Row gutter={16}>
                <Col span={22} offset={1}>
                    <Button onClick={() => { this.props.onSubmit({}) }}>I understand. Start annotating!</Button>
                </Col>
            </Row>
        </>
    }
}


class VideoInstructionsTask extends React.Component {
    render() {
        return <>
            <Row gutter={16} style={{ 'marginTop': '20px' }}>
                <Col span={22} offset={1}>
                    <Title>Video-only ratings</Title>
                    <Paragraph>
                        This task consists in watching a series videos (with no audio) of approximately 10 seconds each, and rating your perception of the requested quantities. 
                    </Paragraph>

                    <Paragraph>
                        Pay attention to the movements of the person and rate the enjoyment according to your own perception of how much they are enjoying their conversation.
                    </Paragraph>

                    <Paragraph>
                        For each video, the process is the following:
                    </Paragraph>
                    <Timeline>
                        <Timeline.Item>The video will start with a frame showing the target person enclosed in a bounding box. Make sure to find the target person before watching the video.</Timeline.Item>
                        <Timeline.Item>Play the video, paying attention to the target person.</Timeline.Item>
                        <Timeline.Item>Give your rating of the requested construct.</Timeline.Item>
                        <Timeline.Item>Click next to go to the next segment.</Timeline.Item>
                    </Timeline>
                </Col>
            </Row>
            <Row gutter={16}>
                <Col span={22} offset={1}>
                    <Button onClick={() => { this.props.onSubmit({}) }}>I understand. Start annotating!</Button>
                </Col>
            </Row>
        </>
    }
}


class AudioInstructionsTask extends React.Component {
    render() {
        return <>
            <Row gutter={16} style={{ 'marginTop': '20px' }}>
                <Col span={22} offset={1}>
                    <Title>Audio-only ratings</Title>
                    <Paragraph>
                        This task consists in listening to a series of audio recordings of a person in conversation, and rating several constructs related to engagement.
                    </Paragraph>

                    <Alert message={'Please make sure that you have working speakers and or headphones. You can test them by making sure that you can listen clearly to the following audio:'}/>

                    <Paragraph>
                        For each audio, the process is the following:
                    </Paragraph>
                    <Timeline>
                        <Timeline.Item>Play the audio, listening to the voices in it. Note that voices of people other than the target speaker may be in the background. Make sure to identify the main speaker (loudest voice).</Timeline.Item>
                        <Timeline.Item>Play the audio, paying attention to the target voice.</Timeline.Item>
                        <Timeline.Item>Give your rating of the requested construct.</Timeline.Item>
                        <Timeline.Item>Click next to go to the next segment.</Timeline.Item>
                    </Timeline>
                </Col>
            </Row>
            <Row gutter={16}>
                <Col span={22} offset={1}>
                    <Button onClick={() => { this.props.onSubmit({}) }}>I understand. Start annotating!</Button>
                </Col>
            </Row>
        </>
    }
}


class AVInstructionsTask extends React.Component {
    render() {
        return <>
            <Row gutter={16} style={{ 'marginTop': '20px' }}>
                <Col span={22} offset={1}>
                    <Title>Audiovisual ratings</Title>
                    <Paragraph>
                        This task consists in watching a series of videos (with audio) of approximately 10 seconds each, and rating several constructs related to engagement.
                    </Paragraph>

                    <Alert message={'Please make sure that you have working speakers and or headphones. You can test them by making sure that you can listen clearly to the following audio example:'} />

                    <Paragraph>
                        For each video, the process is the following:
                    </Paragraph>

                    <Timeline>
                        <Timeline.Item>The video will start with a frame showing the target person enclosed in a bounding box. Make sure to find the target person before watching the video.</Timeline.Item>
                        <Timeline.Item>Play the video, paying attention to the target person.</Timeline.Item>
                        <Timeline.Item>Give your rating of the requested construct.</Timeline.Item>
                        <Timeline.Item>Click next to go to the next segment.</Timeline.Item>
                    </Timeline>
                </Col>
            </Row>
            <Row gutter={16}>
                <Col span={22} offset={1}>
                    <Button onClick={() => { this.props.onSubmit({}) }}>I understand. Start annotating!</Button>
                </Col>
            </Row>
        </>
    }
}

export {
    AudioInstructionsTask,
    VideoInstructionsTask,
    AVInstructionsTask
}