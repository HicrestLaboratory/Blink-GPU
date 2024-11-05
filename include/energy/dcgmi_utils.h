#ifndef DCGMI_UTILS
#define DCGMI_UTILS

#define PICO_HANDLE_DCGMRESULT( R ) {									\
	if ( R != DCGM_ST_OK ) {									\
		fprintf(stderr, "Error: DCGM error in function %s at line %d\n", __func__, __LINE__);	\
		fprintf(stderr, "\t%s\n\n", errorString( R )); 						\
	}												\
}

#define PICO_DCGM_INIT		 		\
	dcgmReturn_t dcgm_result = dcgmInit();	\
	PICO_HANDLE_DCGMRESULT( dcgm_result );


#define PICO_DCGM_FINALIZE dcgmShutdown()

/* ----------------------------------------------------------------------------------------------------------------------------
	dcgmReturn_t dcgmProfGetSupportedMetricGroups(dcgmHandle_t pDcgmHandle, dcgmProfGetMetricGroups_t *metricGroups)
   ----------------------------------------------------------------------------------------------------------------------------

	Get all of the profiling metric groups for a given GPU group.
	Profiling metrics are watched in groups of fields that are all watched together. For instance, if you want to watch DCGM_FI_PROF_GR_ENGINE_ACTIVITY, this might also be in the same group as DCGM_FI_PROF_SM_EFFICIENCY. Watching this group would result in DCGM storing values for both of these metrics.
	Some groups cannot be watched concurrently as others as they utilize the same hardware resource.

*/

/* ----------------------------------------------------------------------------------------------------------------------------
	dcgmReturn_t dcgmProfWatchFields(dcgmHandle_t pDcgmHandle, dcgmProfWatchFields_t *watchFields)
   ----------------------------------------------------------------------------------------------------------------------------

	Request that DCGM start recording updates for a given list of profiling field IDs.
	Once metrics have been watched by this API, any of the normal DCGM field-value retrieval APIs can be used on the underlying fieldIds of this metric group.

   ----------------------------------------------------------------------------------------------------------------------------
	dcgmReturn_t dcgmProfUnwatchFields(dcgmHandle_t pDcgmHandle, dcgmProfUnwatchFields_t *unwatchFields)
   ----------------------------------------------------------------------------------------------------------------------------

	Request that DCGM stop recording updates for all profiling field IDs for all GPUs.

*/

// MG is expect to be a dcgmProfGetMetricGroups_t metricGroups
#define PICO_METRICGROUP_PRINT( MG ) {						\
	printf("-------------------------------\n");				\
	printf("MetricGroup info about %s\n", #MG );				\
	printf("-------------------------------\n\n");				\
	printf("Version: %u\n", MG.version);					\
	printf("GPU Id: %u\n", MG.gpuId);					\
	printf("Metrics (%d):\n", MG.numMetricGroups);				\
	for (int i=0; i< MG.numMetricGroups; i++) {				\
		printf("\t%u.%u:", MG.metricGroups[i].majorId, 			\
				MG.metricGroups[i].minorId);			\
		for (int j=0; j< MG.metricGroups[i].numFieldIds; j++){		\
			printf(" %u", MG.metricGroups[i].fieldIds[j]);		\
		}								\
		printf("\n");							\
	}									\
	printf("-------------------------------\n");				\
}

int list_field_values(unsigned int gpuId, dcgmFieldValue_v1 *values, int numValues, void *userdata)
{
    // The void pointer at the end allows a pointer to be passed to this
    // function. Here we know that we are passing in a null terminated C
    // string, so I can cast it as such. This pointer can be useful if you
    // need a reference to something inside your function.
    std::cout << std::endl;
    std::map<unsigned int, dcgmFieldValue_v1> field_val_map;
    // note this is a pointer to a map.
    field_val_map = *static_cast<std::map<unsigned int, dcgmFieldValue_v1> *>(userdata);

    // Storing the values in the map where key is field Id and the value is
    // the corresponding data for the field.
    for (int i = 0; i < numValues; i++)
    {
        field_val_map[values[i].fieldId] = values[i];
    }


    // Output the information to screen.
    for (std::map<unsigned int, dcgmFieldValue_v1>::iterator it = field_val_map.begin(); it != field_val_map.end();
         ++it)
    {
        std::cout << "Field ID => " << it->first << std::endl;
        std::cout << "Value => ";

        dcgm_field_meta_p field = DcgmFieldGetById((it->second).fieldId);
        unsigned char fieldType = field == nullptr ? DCGM_FI_UNKNOWN : field->fieldType;
        switch (fieldType)
        {
            case DCGM_FT_BINARY:
                // Handle binary data
                break;
            case DCGM_FT_DOUBLE:
                std::cout << (it->second).value.dbl;
                break;
            case DCGM_FT_INT64:
                std::cout << (it->second).value.i64;
                break;
            case DCGM_FT_STRING:
                std::cout << (it->second).value.str;
                break;
            case DCGM_FT_TIMESTAMP:
                std::cout << (it->second).value.i64;
                break;
            default:
                std::cout << "Error in field types. " << (it->second).fieldType << " Exiting.\n";
                // Error, return > 0 error code.
                return 1;
                break;
        }

        std::cout << std::endl;
        // Shutdown DCGM fields. This takes care of the memory initialized
        // when we called DcgmFieldsInit.
    }

    // Program executed correctly. Return 0 to notify DCGM (callee) that it
    // was successful.
    return 0;
}

#endif
