class BusinessWav2Lip(BusinessBase):
    def __init__(self):
        super().__init__()
        self.can_run = False
        # 当前场景id
        self.current_scene_id = ''
        # 当前视频素材帧
        self.current_video_section_frame = 1
        # 当前视频素材播放方向：True为正，False为反
        self.current_video_play_direction = True
        self.temp_result = []


    async def wait_wav2lip_result(self, params):

        time_diff= performance_tools.TimeDiff()
        try:
            self.parent.wav2lip_request_queue.put(params)
            start_cnt = time.perf_counter()
            output_resutl = self.parent.wav2lip_result_queue.get()

        except Exception:
            start_cnt = time.perf_counter()
            output_resutl = None
            logger.error(f'wait_wav2lip_result:{traceback.format_exc()}')
        audio_type = "静默"
        if not params[1].startswith("silent_"):
            audio_type = "说话"
        logger.info(f"wav2lp_time :{audio_type}- {params[1]} {time_diff.diff_last('从推送到接收数据耗时')}  time cost on waiting for {time.perf_counter() - start_cnt}")
        return output_resutl

    async def gen_video(self, speech: Speech, cur_node: Node):
        logger.info(f'[{speech.speech_id}] gen_video ing')

        speech.is_wav2lip = True
        try:
            duration = 0
            # TODO 后面删除下面的逻辑，不需要duration 逻辑
            last_node = self.parent.runtime_data.current_node
            while last_node is not None:
                if last_node.data.priority > 0:
                    # priority_node = last_node
                    duration += last_node.data.duration
                else:
                    break
                last_node = last_node.tail
            # 上一个node不说话，当前node 说话 
            logger.debug(f"{speech.speech_id} cur_node.tail: {cur_node.tail }")

            if cur_node.tail and "silent" not in cur_node.tail.data.speech_id:
                before_speak = True
                logger.debug(f"{speech.speech_id} before_speak:{before_speak}")
            else:
                before_speak = False

            # 记录下是首次开始讲话,  如果接着两句都是在讲话，认为一直在讲话,first_speak=False
            # 只要上句在讲话，就认为继续讲话
            if self.parent.last_batch_speak:
                first_speak = False
            else:
                first_speak = False if not speech.first_speak else True
            
            # 如果下句是静默。一定是讲话结束
            if cur_node.tail:
                logger.debug(f"cur_speech: {speech.speech_id}, cur_node.tail.data.speech_id：{cur_node.tail.data.speech_id}")
            if "silent" not in cur_node.data.speech_id and cur_node.tail and "silent" in cur_node.tail.data.speech_id:
                last_speak = True
            else:
                last_speak = False
            logger.debug(f"speech_id:{speech.speech_id}, final_first_speak:{first_speak}, priority:{speech.priority},first_speak:{speech.first_speak} ,last_speak:{last_speak},last_batch_speak:{self.parent.last_batch_speak}")
            params = [
                int(speech.priority == 0),
                speech.speech_id,
                speech.video_model_info.pkl_url,
                self.current_video_section_frame,
                self.current_video_play_direction,
                duration,
                False,
                first_speak,
                last_speak,
                before_speak
            ]

            if self.parent.data_cache:
                self.parent.data_cache.put_data(speech.push_unit_data[0], speech.push_unit_data[1], speech.speech_id)

                diff = time.time()
                result = await self.wait_wav2lip_result(params)
                logger.debug(f'wait_wav2lip_result:{time.time() - diff} ')
                self.current_video_section_frame = result[1]
                self.current_video_play_direction = result[2]

                logger.info(
                    f'[{speech.speech_id}] gen_video succeed:{len(speech.push_unit_data[1])}')
            # 真正放在任务队列中的顺序来判定, 额外有个判定条件当生成速度很快时如果仍然是两段连续的话，应该在等待期间将 self.parent.last_batch_speak设置成False
            # 更新状态
            if "silent" not in speech.speech_id and not self.parent.runtime_data.stop_generate:
                self.parent.last_batch_speak = True
            else:
                self.parent.last_batch_speak = False
            
            self.parent.last_node_silent = True if "silent" in speech.speech_id else False
            logger.info(f"set speechid {speech.speech_id }:{self.parent.last_node_silent}, cause {'silent' in speech.speech_id}")
            
        except Exception:
            logger.error(f'[{speech.speech_id}] gen_video error:{traceback.format_exc()}')
        speech.video_url = 'success'
        speech.is_wav2lip = False

    async def first_run(self):
        while self.parent.is_valid():
            if self.parent.runtime_data.speech_linked_list.head is not None:
                return True
            await asyncio.sleep(tick_time)
        return False

    async def run(self):
        if not await self.first_run():
            logger.warning('exit wav2lip logic at begin')
            return
        logger.info('start wav2lip looping')

        while self.parent.is_valid():
            logger.debug('run_time_cal begin')
            # logger.debug('run')
            if self.parent.runtime_data.current_node is None:
                self.parent.runtime_data.current_node = self.parent.runtime_data.speech_linked_list.head
                logger.warning(f'当前 self.parent.runtime_data.current_node 节点为None, turn to {self.parent.runtime_data.speech_linked_list.head.data.speech_id}')
            last_node = self.parent.runtime_data.current_node

            # unplayed_duration = self.parent.runtime_data.get_unplayed_duration()   
            # logger.debug(f'unplayed_duration {unplayed_duration} ')
            is_write_fast = False and self.parent.is_write_fast()
            # is_write_fast = self.parent.is_write_fast()
            is_read_fast = self.parent.is_read_fast()


            while last_node is not None:
                if not self.parent.is_valid():
                    logger.warning('exit wav2lip logic')
                    return

                # -------- 首次缓冲逻辑 first_wav2lip_ok逻辑  begin
                if not self.parent.runtime_data.first_wav2lip_ok:
                    wav2lip_full_duration = 0
                    calculate_node = self.parent.runtime_data.current_node
                    while calculate_node is not None:
                        if calculate_node.data.video_url != '':
                            wav2lip_full_duration += calculate_node.data.duration
                        else:
                            break
                        calculate_node = calculate_node.tail
                    if wav2lip_full_duration >= config.g_cache_first_wav2lip_duration:
                        logger.warning('first_wav2lip_ok')
                        self.parent.runtime_data.first_wav2lip_ok = True
                # -------- 首次缓冲逻辑 first_wav2lip_ok逻辑  end

                if not self.parent.wav2lip_process_started:
                    break
                # 如果共享内存实际读取与写入的缓冲不足
                # 或者 liveroom计算的剩余缓冲不足
                # 或者 开播缓冲不足, 这个逻辑目前去掉了
                # 会重新准备音频数据
                if not is_write_fast or is_read_fast or not self.parent.runtime_data.first_wav2lip_ok :
                    # 如果当前节点未推送 并且 未生成wav2lip 并且 已准备好音频数据
                    if ((not last_node.data.had_pushed) and last_node.data.video_url == '' and last_node.data.audio_url != ''):
                        await self.gen_video(last_node.data, last_node)
                        break
                    last_node = last_node.tail
                    self.parent.runtime_data.stop_generate = False
                else:
                    self.parent.runtime_data.stop_generate = True
                    break
            await asyncio.sleep(tick_time)

        logger.warning('exit wav2lip logic')