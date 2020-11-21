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


class VideoInstructionsTask extends React.Component {
    render() {
        return <>
            <Row gutter={16} style={{ 'marginTop': '20px' }}>
                <Col span={22} offset={1}>
                    <Title>Video detection of laughter</Title>
                    <Paragraph>
                        This task consists in watching 20 videos of approximately 10 seconds each, and rating the enjoyment of the marked target person.
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

class AudiovisualInstructionsTask extends React.Component {
    state = {
        captcha: '',
        error: null
    }

    handleSubmit = () => {
        if(this.state.captcha == '1367') this.props.onSubmit({})
        else this.setState({error: 'Incorrect numbers. Please try again.'})
    }

    onCaptchaChange = (e: Event) => {
        this.setState({
            captcha: e.target.value
        })
    }

    render() {
        return <>
            <Row gutter={16} style={{'marginTop': '20px'}}>
                <Col span={22} offset={1}>
                    <Title>Video detection of laughter</Title>
                    <Paragraph>
                        This task consists in watching 20 videos of approximately 10 seconds each, and rating the enjoyment of the marked target person.
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
                    <p>Please play the following audio and input the numbers you hear into the box to continue:</p>

                    <p><Input value={this.state.captcha} onChange={this.onCaptchaChange} placeholder='Numbers' style={{ width: '200px' }} /></p>
                    
                    {this.state.error ? <Alert message={this.state.error} type="error" showIcon /> : ''}

                    <Button type="primary" onClick={this.handleSubmit}>Start annotating!</Button>
                </Col>
            </Row>
        </>
    }
}

class LocalVideoInstructionsTask extends React.Component {
    render() {
        return <>
            <Row gutter={16} style={{ 'marginTop': '20px' }}>
                <Col span={22} offset={1}>
                    <Title>Video detection of laughter</Title>
                    <Paragraph>
                        This task consists in watching 20 videos of approximately 10 seconds each, and rating the enjoyment of the marked target person.
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

class LocalAudioInstructionsTask extends React.Component {
    render() {
        return <>
            <Row gutter={16} style={{ 'marginTop': '20px' }}>
                <Col span={22} offset={1}>
                    <Title>Video detection of laughter</Title>
                    <Paragraph>
                        This task consists in watching 20 videos of approximately 10 seconds each, and rating the enjoyment of the marked target person.
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

class LocalAudiovisualInstructionsTask extends React.Component {
    render() {
        return <>
            <Row gutter={16} style={{ 'marginTop': '20px' }}>
                <Col span={22} offset={1}>
                    <Title>Video detection of laughter</Title>
                    <Paragraph>
                        This task consists in watching 20 videos of approximately 10 seconds each, and rating the enjoyment of the marked target person.
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
                    <Button onClick={()=>{this.props.onSubmit({})}}>I understand. Start annotating!</Button>
                </Col>
            </Row>
        </>
    }
}
export {
    VideoInstructionsTask,
    LocalVideoInstructionsTask,
    LocalAudioInstructionsTask,
    LocalAudiovisualInstructionsTask
}